import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable

from openai.types.chat import ChatCompletionMessageParam

from .context import ContextCompactor
from .events import ContentDelta, ContentDone, ReasoningDelta, UsageEvent
from .executor import ToolExecutor
from .memory import MemoryManager
from .permission import PermissionResolver
from .llm import LLMClient
from .prompt import build_system_prompt
from .renderer import Renderer
from .session import SessionState, UsageRecord
from .protocol import TurnAbort, parse_turn
from .tools.base import Tool
from .util import build_tool_results_message, estimate_message_tokens
from .verifier import Verifier

if TYPE_CHECKING:
    from .checkpoint import SessionCheckpointStore


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        session_state: SessionState,
        renderer: Renderer,
        tool_timeout: float = 30,
        context_watermark: float = 0.75,
        keep_recent_tool_results: int = 3,
        max_consecutive_invalid: int = 3,
        permission_resolver: PermissionResolver | None = None,
        cancellation_check: Callable[[], bool] | None = None,
        memory: MemoryManager | None = None,
        verifier: Verifier | None = None,
        max_verification_retries: int = 3,
        checkpoint_store: "SessionCheckpointStore | None" = None,
    ):
        self.llm = llm
        self.session_state = session_state
        self.renderer = renderer
        # 长期记忆协作者:只主 Agent 注入,子 Agent 传 None(保持纯净隔离上下文)。
        # Agent 只在主循环里喊它三声:构造时取指令、每轮注入召回、收口后提取落盘。
        self.memory = memory
        self.verifier = verifier
        self.max_verification_retries = max_verification_retries
        self.checkpoint_store = checkpoint_store
        self.last_checkpoint_error: Exception | None = None
        if max_verification_retries < 1:
            raise ValueError("max_verification_retries 必须 >= 1")
        # 权限裁决器可由装配层注入(承载规则/模式配置),并沿主→子 Agent 共用同一份;
        # 不传则 ToolExecutor 自建一个无 handler 的默认 resolver(ask 一律 fail-closed)。
        self._permission_resolver = permission_resolver
        self._cancellation_check = cancellation_check
        # 连续 N 轮解析失败就止损:再喂回去也大概率是同样的废 JSON,
        # 与其烧光 max_steps,不如如实标 failed 退出。中间成功一次即清零。
        self.max_consecutive_invalid = max_consecutive_invalid

        # 上下文压缩独立成 collaborator:Agent 只负责在主循环里喊它一声 +
        # 折叠后从 running total 扣减省下的 token,折叠逻辑本身归 ContextCompactor。
        self.compactor = ContextCompactor(
            renderer,
            context_watermark=context_watermark,
            keep_recent_tool_results=keep_recent_tool_results,
        )

        if not self.session_state.message_records:
            # 有记忆时把静态记忆指令段拼进 system prompt(类型分类法/如何保存/何时存取/
            # 据记忆行动前先核实)。MEMORY.md 内容和相关记忆不在这里——走每轮注入保新鲜。
            memory_section = self.memory.instructions() if self.memory else ""
            msg: ChatCompletionMessageParam = {
                "role": "system",
                "content": build_system_prompt(
                    json.dumps(
                        [tool.to_dict() for tool in tools], ensure_ascii=False, indent=2
                    ),
                    memory_section=memory_section,
                ),
            }
            self.session_state.append_message(msg)

        # 工具调度执行独立成 collaborator:Agent 只在主循环里把这一轮的 tool_calls
        # 交给它,查表/钳超时/并发分流/异常兜底都归 ToolExecutor。
        # registry 存整个 Tool:执行要 call,调度要 concurrency 等元数据。
        self.executor = ToolExecutor(
            {tool.name: tool for tool in tools},
            tool_timeout=tool_timeout,
            on_command_output=renderer.on_command_output,
            permission_resolver=permission_resolver,
            session_state=session_state,
            cancellation_check=cancellation_check,
        )

    @property
    def context_limit(self) -> int | None:
        return self.llm.context_limit

    @property
    def messages(self) -> Sequence[ChatCompletionMessageParam]:
        return self.session_state.messages

    def _compact_context_if_needed(self, transient_tokens: int = 0) -> int:
        """喊 compactor 折叠旧工具结果;折叠后从 running total 扣减省下的 token。

        不再作废锚点:running total 被增量调整(减去折叠省下的),
        下次 usage 回来时自然会精确校准。
        """
        folded_count, token_savings = self.compactor.compact_if_needed(
            self.session_state.message_records,
            self.session_state.context_tokens + transient_tokens,
            self.context_limit,
        )
        if token_savings:
            self.session_state.context_tokens -= token_savings
        return folded_count

    def _run_turn(
        self, plan_reminder: ChatCompletionMessageParam | None = None
    ) -> tuple[str, UsageRecord | None]:
        """跑一轮 LLM 调用：实时渲染事件流，返回拼接好的完整 content。"""

        # 初始化空串:依赖"LLMClient 必以 ContentDone 收尾"的契约,
        # 但契约被破坏时不该炸出莫名其妙的 NameError
        content = ""
        usage_record: UsageRecord | None = None

        wire_messages = self.session_state.wire_messages()
        # 当前计划是运行期状态，不永久复制进 transcript。每轮临时放在 wire 尾部，
        # 即使旧的 create/update 工具结果日后被 context compactor 折叠，模型仍能看到
        # 最新计划；无计划时完全不改变原消息流。
        if plan_reminder is not None:
            wire_messages.append(plan_reminder)

        for event in self.llm(wire_messages):
            if isinstance(event, ReasoningDelta):
                self.renderer.on_reasoning_delta(event.piece)
            elif isinstance(event, ContentDelta):
                self.renderer.on_content_delta(event.piece)
            elif isinstance(event, ContentDone):
                content = event.content
            elif isinstance(event, UsageEvent):
                usage_record = UsageRecord.from_usage(event.usage)

                self.renderer.on_usage(
                    usage_record.prompt_tokens,
                    usage_record.completion_tokens,
                    usage_record.total_tokens,
                    self.context_limit,
                )

        return content, usage_record

    def _plan_reminder(self) -> ChatCompletionMessageParam | None:
        block = self.session_state.plan_manager.to_prompt_block()
        # 计划字段由模型工具调用产生，最终也可能来自不可信用户文本；保持 user role，
        # 并由 to_prompt_block 的 JSON 数据边界明确它不具备指令权限。
        return {"role": "user", "content": block} if block else None

    def _record_usage_for_turn(
        self,
        turn_record,
        usage_record: UsageRecord,
        transient_plan_tokens: int,
    ) -> None:
        self.session_state.record_usage_for_turn(turn_record, usage_record)
        # 服务端 prompt_tokens 包含临时计划提醒，但它没有落进 transcript；扣除其估算值，
        # 令 running total 始终表示持久 transcript。下一轮压缩时会重新加上最新提醒。
        self.session_state.context_tokens = max(
            0, self.session_state.context_tokens - transient_plan_tokens
        )

    def run(self, prompt: str, max_steps: int | None = None) -> str | None:
        """执行新任务。"""
        max_steps = self.session_state.max_steps if max_steps is None else max_steps
        if max_steps <= 0:
            raise ValueError("max_steps 必须 > 0")
        self.session_state.max_steps = max_steps
        # Plan 是 user-turn 级状态，不是跨任务记忆。首轮允许装配层预置计划；
        # 后续 run() 开始新目标时清空旧计划，continue_run() 则原样保留。
        if self.session_state.turns:
            self.session_state.plan_manager.reset()
        # 重置上一轮的终态,使 status 始终反映"当前这轮"(多轮 REPL 下尤其需要)。
        self.session_state.mark_running()
        self.session_state.begin_user_turn(prompt)
        self.session_state.append_message({"role": "user", "content": prompt})

        # 自动召回:针对本轮 prompt 选出相关记忆 + MEMORY.md 索引,作为 system-reminder
        # 注入(role=user 以兼容各端点)。走 append_message 自动计入 context_tokens。
        # 召回是尽力而为的旁路,内部已吞异常,空块则跳过。
        if self.memory:
            recall_block = self.memory.recall_block(prompt)
            if recall_block:
                self.session_state.append_message(
                    {"role": "user", "content": recall_block}
                )

        self._checkpoint()

        return self._run_loop(max_steps)

    def continue_run(self, max_steps: int | None = None) -> str | None:
        """Continue a running session loaded from a checkpoint.

        This is intentionally separate from ask_user: human interaction remains
        synchronous inside the permission layer and has no pause/resume state.
        """
        if self.session_state.status != "running":
            raise ValueError(
                f"只能继续 status=running 的会话，当前为 {self.session_state.status}"
            )
        budget = self.session_state.max_steps if max_steps is None else max_steps
        if budget <= 0:
            raise ValueError("max_steps 必须 > 0")
        self._checkpoint()
        return self._run_loop(budget)

    def _checkpoint(self) -> None:
        if self.checkpoint_store is None:
            return
        try:
            self.checkpoint_store.save(self.session_state)
            self.last_checkpoint_error = None
        except Exception as exc:
            # Persistence is a reliability sidecar: surface the failure for
            # callers/tests, but do not destroy an otherwise valid agent turn.
            self.last_checkpoint_error = exc
            self.renderer.on_checkpoint_error(f"{type(exc).__name__}: {exc}")

    def _finalize_memory(
        self, final_answer: str | None, *, extract_semantic: bool
    ) -> None:
        if self.memory is not None:
            self.memory.finalize_turn(
                self.session_state,
                final_answer,
                extract_semantic=extract_semantic,
            )

    def _run_loop(
        self,
        max_steps: int,
        consecutive_invalid: int = 0,
        consecutive_verification: int = 0,
    ) -> str | None:
        """运行当前 user turn 直到 final_answer 或耗尽步数。"""

        for loop_index in range(max_steps):
            if self._cancellation_check and self._cancellation_check():
                self.session_state.mark_failed()
                self._finalize_memory(None, extract_semantic=False)
                self._checkpoint()
                return None
            plan_reminder = self._plan_reminder()
            transient_plan_tokens = (
                estimate_message_tokens(plan_reminder) if plan_reminder else 0
            )
            self._compact_context_if_needed(transient_plan_tokens)

            # ----- 步骤 1：调用 LLM 推理 -----
            content, usage_record = self._run_turn(plan_reminder)
            # assistant 原文不再在这里手动入队:改由 session 的 record_* 方法
            # 在记账的同时落进 wire,wire 与 turn 原子产生、靠稳定 id 关联。

            # ----- 步骤 2：解析 + 校验(协议层) -----
            # parse_turn 把"解析 JSON + 校验形状 + 二选一路由 + 解析 tool_calls"
            # 一次性收口在 protocol 层;形状级错误统一抛 TurnAbort,主循环只管分流。
            try:
                turn = parse_turn(content)
                consecutive_invalid = 0  # 解析成功,连击清零

                if turn.kind == "final":
                    # 先落候选 turn，再运行 completion gate。若被拒绝，它仍是有价值的
                    # 历史证据，验证反馈会作为下一条 user message 进入正常 ReAct 循环。
                    turn_record = self.session_state.record_assistant_turn(
                        assistant_raw=content,
                        parsed=turn.parsed,
                        route="final",
                    )
                    if usage_record is not None:
                        self._record_usage_for_turn(
                            turn_record, usage_record, transient_plan_tokens
                        )

                    verification = (
                        self.verifier.verify(self.session_state, turn.final_answer)
                        if self.verifier
                        else None
                    )
                    if verification is not None:
                        self.session_state.record_verification(
                            turn_record,
                            verification.approved,
                            [issue.to_dict() for issue in verification.issues],
                        )
                        if not verification.approved:
                            consecutive_verification += 1
                            self.session_state.append_message(
                                verification.feedback_message()
                            )
                            self._checkpoint()
                            if (
                                consecutive_verification
                                >= self.max_verification_retries
                            ):
                                self.renderer.on_final(
                                    "最终答案连续未通过完成验证，任务终止。"
                                )
                                self.session_state.mark_failed()
                                self._finalize_memory(
                                    None, extract_semantic=False
                                )
                                self._checkpoint()
                                return None
                            continue

                    self.renderer.on_final(turn.final_answer)
                    self.session_state.mark_completed()
                    # 每个终态都记录 episode；只有成功回合才提取长期语义记忆。
                    self._finalize_memory(
                        turn.final_answer, extract_semantic=True
                    )
                    self._checkpoint()
                    return turn.final_answer

                if turn.kind == "tool_calls":
                    # 更新会话，添加成功回合记录
                    turn_record = self.session_state.record_assistant_turn(
                        assistant_raw=content,
                        parsed=turn.parsed,
                        route="tool_calls",
                        tool_calls=turn.tool_calls,
                    )
                    if usage_record is not None:
                        self._record_usage_for_turn(
                            turn_record, usage_record, transient_plan_tokens
                        )

                    # 工具副作用前先落 pending checkpoint。若进程在调用期间崩溃，
                    # 恢复层会把这些调用标成 outcome unknown 并要求模型先检查现场，
                    # 不会把同一个写操作静默重放。
                    self._checkpoint()

                    outcomes = self.executor.execute(
                        turn.tool_calls,
                        on_call=self.renderer.on_tool_call,
                        on_result=self.renderer.on_tool_result,
                    )

                    for outcome in outcomes:
                        self.session_state.record_tool_execution(
                            call_id=outcome.call.id,
                            result=outcome.result,
                            status=outcome.status,
                        )

                    self.session_state.append_message(
                        build_tool_results_message(
                            [
                                (outcome.call, outcome.result)
                                for outcome in outcomes
                            ]
                        )
                    )
                    self._checkpoint()

            except TurnAbort as e:
                consecutive_invalid += 1

                # 更新会话，添加失败回合记录
                turn_record = self.session_state.record_invalid_turn(
                    content,
                    f"LLM 输出无法解析或路由: {e}",
                )
                if usage_record is not None:
                    self._record_usage_for_turn(
                        turn_record, usage_record, transient_plan_tokens
                    )

                # 连续失败到阈值就止损:再喂回去多半还是同样的废 JSON。
                if consecutive_invalid >= self.max_consecutive_invalid:
                    self.renderer.on_final(
                        f"连续 {consecutive_invalid} 轮输出无法解析，任务终止。"
                    )
                    self.session_state.mark_failed()
                    self._finalize_memory(None, extract_semantic=False)
                    self._checkpoint()
                    return None

                # 没到阈值:把错误喂回模型,给它一次改正的机会
                self.session_state.append_message({
                    "role": "user",
                    "content": json.dumps(
                        {"error": f"LLM 输出无法解析或路由：{e}"},
                        ensure_ascii=False,
                    ),
                })
                self._checkpoint()
                continue

        else:
            self.renderer.on_final(
                f"已达到最大步数上限（{max_steps} 步），任务未完成。"
            )
            self.session_state.mark_max_steps()
            self._finalize_memory(None, extract_semantic=False)
            self._checkpoint()
            return None
