"""
ReActMulti Agent 主入口模块（多工具版）

和隔壁 ReAct 的唯一区别：一个回合可以发起【多个】工具调用。
单工具版是严格串行 think→act(1个)→observe；这一版是 think→act(N个)→observe(N个)。

"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit import PromptSession, prompt

from .agent import Agent
from .checkpoint import CheckpointError, SessionCheckpointStore
from .logger import get_logger
from .memory import MemoryManager
from .tools import tools as base_tools
from .tools.ask_user_tool import ask_user_tool
from .llm import LLMClient
from .permission import (
    FallbackApprovalHandler,
    InteractiveApprovalHandler,
    PermissionCheckResult,
    PermissionRequest,
    PermissionResolver,
    RuleBasedApprovalHandler,
    append_allow_rule,
    load_permission_settings,
)
from .renderer import ConsoleRenderer
from .session import SessionState
from .subagent import build_agent_tools
from .tools.mcp_client import McpManager, load_mcp_config
from .verifier import Verifier


logger = get_logger(__name__)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReActMulti interactive agent")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        nargs="?",
        const="",
        metavar="SESSION_ID",
        help="从指定 session checkpoint 恢复 (留空则列出历史菜单选择)",
    )
    resume_group.add_argument(
        "-c",
        "--continue",
        dest="continue_latest",
        action="store_true",
        help="恢复最近保存的 session checkpoint",
    )
    parser.add_argument(
        "--no-session-persistence",
        action="store_true",
        help="本次运行不保存 checkpoint",
    )
    return parser.parse_args()


def _make_interaction_handler(renderer: ConsoleRenderer):
    """ask_user 的交互 adapter 工厂：UI 委托给 renderer，这里只做「原始回答 → PermissionCheckResult」的翻译。"""

    def handler(request: PermissionRequest) -> PermissionCheckResult:
        arguments = request.arguments
        answer = renderer.prompt_user(
            question=arguments["question"].strip(),
            context=arguments.get("context", "").strip(),
            options=tuple(arguments.get("options") or ()),
        )
        if answer is None:
            return PermissionCheckResult(
                "deny",
                "用户取消回答问题",
                request.check.risk_flags,
                source="user_interaction",
            )
        return PermissionCheckResult(
            "allow",
            "用户已回答问题",
            request.check.risk_flags,
            updated_arguments={**arguments, "answer": answer},
            source="user_interaction",
        )

    return handler



if __name__ == "__main__":
    args = parse_cli_args()
    load_dotenv()

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    context_limit_raw = os.getenv("OPENAI_CONTEXT_LIMIT")
    context_limit = int(context_limit_raw) if context_limit_raw else None

    llm_client = LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        context_limit=context_limit or 128000,
    )

    # 记忆的召回/提取 side-query 可选用更便宜的模型省钱(对标 memdir 用 Sonnet 选记忆)。
    # 配了 OPENAI_MEMORY_MODEL 就单独建个非流式 client,否则复用主 client。
    memory_model = os.getenv("OPENAI_MEMORY_MODEL")
    selector_llm = (
        LLMClient(
            base_url=base_url,
            api_key=api_key,
            model=memory_model,
            stream=False,
        )
        if memory_model
        else llm_client
    )

    verifier_model = os.getenv("OPENAI_VERIFIER_MODEL")
    verifier_llm = (
        LLMClient(
            base_url=base_url,
            api_key=api_key,
            model=verifier_model,
            stream=False,
        )
        if verifier_model
        else selector_llm
    )

    renderer = ConsoleRenderer()

    workspace_dir = Path(__file__).resolve().parent / "workspace"
    # 多轮对话:session 整段存活,每轮把用户输入 append 进同一条历史；Agent.run 会把
    # user_goal 更新为当前任务，供 Verifier 和 checkpoint 使用。
    checkpoint_store = SessionCheckpointStore(workspace_dir / ".react_sessions")
    resumed = bool(args.resume is not None or args.continue_latest)
    try:
        if args.resume is not None:
            session_id = args.resume.strip()
            if not session_id:
                recent = checkpoint_store.list_recent_sessions(limit=5)
                if not recent:
                    raise CheckpointError("没有找到任何可恢复的历史 checkpoint")
                print("\n请选择要恢复的历史会话：")
                for i, item in enumerate(recent, 1):
                    goal = item["user_goal"] or "(无目标描述)"
                    if len(goal) > 40:
                        goal = goal[:37] + "..."
                    print(
                        f"  [{i}] {item['saved_at']} ({item['session_id']}) "
                        f'| "{goal}" (status: {item["status"]})'
                    )
                print()
                choice_str = prompt("输入序号 (默认 [1]): ").strip()
                idx = 0
                if choice_str:
                    try:
                        idx = int(choice_str) - 1
                    except ValueError:
                        raise CheckpointError(f"无效的选择: {choice_str}")
                if not (0 <= idx < len(recent)):
                    raise CheckpointError(f"选择超出范围: {choice_str}")
                session_id = recent[idx]["session_id"]
            session_state = checkpoint_store.load(session_id)
        elif args.continue_latest:
            session_state = checkpoint_store.load_latest()
        else:
            session_state = SessionState.create(
                user_goal="(interactive session)",
                workspace_dir=workspace_dir,
            )
    except CheckpointError as exc:
        raise SystemExit(f"无法恢复会话: {exc}") from exc


    # MCP 接入:从 workspace 下的 .mcp.json 发现外部 stdio server,连接并把它们的工具
    # 翻译成本系统的 Tool。session 由 mcp_manager 持有,整段运行期保持存活,finally 关闭。
    # 没配 .mcp.json 时 configs 为空,start() 直接返回 [],对其余流程完全无感。
    mcp_manager = McpManager(load_mcp_config(workspace_dir / ".mcp.json"))
    mcp_tools = mcp_manager.start()

    # 权限裁决:加载持久化配置(模式 + allow/deny 规则),按"要不要人"两种装配。
    #
    # 要不要人,默认看有没有真终端,不用记环境变量(env 仍可强制覆盖):
    #   - 有 TTY(你坐在终端前) → 规则 + 人:规则 on_no_match=ask 对灰色地带"弃权",
    #     落到交互式 handler 弹窗问你;rm/sudo 等 deny 仍直接拒、不打扰你。
    #   - 无 TTY(管道/CI/后台) → 纯规则,on_no_match=deny 直接 fail-closed,绝不阻塞。
    # 关键:能不能被问到,取决于规则有没有提前 allow 它——allow 列得越全,落到人手里越少。
    # 默认配置只 allow 只读命令,所以写文件/网络/python 都会落到你这来确认。
    # 主 Agent 与所有子 Agent 共用这同一份 resolver,规则/记忆全树一致。
    settings = load_permission_settings()
    env_interactive = os.getenv("REACT_PERMISSION_INTERACTIVE")
    interactive = (
        env_interactive == "1"
        if env_interactive is not None
        else sys.stdin.isatty()
    )
    if interactive:
        approval_handler = FallbackApprovalHandler(
            RuleBasedApprovalHandler(settings, on_no_match="ask"),
            # on_remember:用户选"别再问"时把规则写回 settings.json,下次同工具在规则层
            # 就自动放行(连这个交互 handler 都到不了)——对标 Claude Code 的"Yes, don't ask again"。
            InteractiveApprovalHandler(renderer=renderer, on_remember=append_allow_rule),
        )
    else:
        approval_handler = RuleBasedApprovalHandler(settings)
    permission_resolver = PermissionResolver(
        approval_handler=approval_handler,
        interaction_handler=_make_interaction_handler(renderer) if interactive else None,
    )

    # 给主 Agent 装上"基础工具 + spawn_agent"的分层工具集:depth=0 是主 Agent,
    # 它能委派出 depth=1 的子 Agent;到 max_depth 那层不再带 spawn,递归到底。
    tools = build_agent_tools(
        llm_client,
        base_tools + mcp_tools,
        depth=0,
        max_depth=2,
        permission_resolver=permission_resolver,
    )

    # ask_user 与记忆工具只给主 Agent:都在 build_agent_tools 之后【单独追加】，不进
    # base_tools。子 Agent 不能绕过父 Agent 直接打断人；它若信息不足，应把缺口作为
    # 结果交回父 Agent，由父 Agent 决定是否询问。子 Agent 也保持无长期记忆的纯净上下文。
    memory_manager = MemoryManager(llm_client, selector_llm=selector_llm)
    tools = [*tools, ask_user_tool, *memory_manager.tools()]

    agent = Agent(
        llm_client,
        tools,
        session_state,
        renderer,
        keep_recent_tool_results=3,
        permission_resolver=permission_resolver,
        memory=memory_manager,
        verifier=Verifier(verifier_llm),
        checkpoint_store=(None if args.no_session_persistence else checkpoint_store),
    )

    # ---- REPL：外层多轮循环 ----
    # 对标 Claude Code 的 REPL：内层 agent.run() 把【一个 user turn】跑到 final_answer
    # 就交还控制权;外层在这里读下一句输入,复用【同一个 agent / session】再 run。
    # 历史天然续上。ask_user 会在工具执行期间直接阻塞等待用户输入，不需要退出/恢复主循环。
    # 退出:Ctrl-D / Ctrl-C / 输入 /exit | /quit。
    try:
        if resumed:
            print(
                f"已恢复 session {session_state.session_id} "
                f"(status={session_state.status})"
            )
            if session_state.status == "running":
                agent.continue_run()
        prompt_session: PromptSession[str] = PromptSession()
        while True:
            user_input = renderer.prompt_main_input(prompt_session)
            if user_input is None or user_input in ("/exit", "/quit"):
                break
            if not user_input:
                continue
            agent.run(user_input)
        if agent.checkpoint_store:
            print(f"💾 会话已保存 (session_id: {session_state.session_id})")
    finally:
        # 关闭 MCP session / stdio 子进程,避免残留进程。
        mcp_manager.shutdown()
