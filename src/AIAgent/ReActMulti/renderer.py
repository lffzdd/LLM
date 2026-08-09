"""UI 前端层：所有人机交互的唯一出入口。

职责分两组：
1. **单向输出**（on_* 回调）：Agent 生命周期事件的展示——思考、工具、答案、用量等。
2. **双向交互**（prompt_* 方法）：需要阻塞等待用户输入的场景——权限确认、ask_user 问答。

主循环只跟 Renderer 接口打交道，不关心具体怎么展示/收集输入。
这样同一套 Agent 逻辑可以配不同的 Renderer：

    ConsoleRenderer  → 终端实时输出 + 终端交互（当前默认）
    SilentRenderer   → 什么都不打（跑测试 / 批量任务）
    （未来）JSONRenderer / WebRenderer → 把事件推给前端

交互方法在基类提供 fail-closed 默认实现（拒绝/返回 None），
不支持交互的渲染器（Silent/SubAgent）无需覆盖。
"""

import json
import sys
import threading
from abc import ABC, abstractmethod
from typing import Any

from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .tools.base import ToolCall, ToolResult


class Renderer(ABC):
    """UI 前端接口。主循环按 ReAct 的生命周期回调 on_* 方法展示事件，
    按需调用 prompt_* 方法进行双向交互。"""

    @abstractmethod
    def on_reasoning_delta(self, piece: str) -> None: ...
    @abstractmethod
    def on_content_delta(self, piece: str) -> None: ...
    @abstractmethod
    def on_tool_call(self, tool_call: ToolCall | dict) -> None: ...
    @abstractmethod
    def on_tool_result(self, tool_result: "ToolResult | dict") -> None: ...
    @abstractmethod
    def on_final(self, answer: Any) -> None: ...

    def on_usage(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        context_limit: int | None,
    ) -> None:
        """本轮 token 用量回调(服务端精确值)。默认不输出，子类按需覆盖。"""

    def on_context_compact(
        self,
        folded_count: int,
        prompt_tokens: int | None,
        context_limit: int | None,
        context_watermark: float,
    ) -> None:
        """上下文压缩回调。默认不输出，子类按需覆盖。"""

    def on_command_output(self, line: str) -> None:
        """命令流式输出回调。默认不输出，子类按需覆盖。"""

    def on_checkpoint_error(self, error: str) -> None:
        """Checkpoint 持久化失败。默认不输出，交互渲染器应明确告警。"""

    # ── 双向交互（子类按能力覆盖，默认 fail-closed） ──

    def prompt_permission(
        self,
        tool_name: str,
        subject: str,
        risk_flags: str,
        reason: str,
        offer_always: bool,
    ) -> str:
        """展示权限确认请求并收集用户选择，返回原始输入字符串。

        默认实现直接返回 ``"n"``（fail-closed），不支持交互的渲染器
        （SilentRenderer / SubAgentRenderer）继承此默认即可。
        """
        return "n"

    def prompt_user(
        self,
        question: str,
        context: str = "",
        options: tuple[str, ...] = (),
    ) -> str | None:
        """展示 ask_user 问题并阻塞等待用户回答。

        返回用户输入的非空字符串；返回 ``None`` 表示用户取消（Ctrl-C/EOF）
        或渲染器不支持交互。默认实现返回 ``None``。
        """
        return None

    def prompt_main_input(self, prompt_session: Any = None) -> str | None:
        """主 REPL 循环展示输入提示并等待用户指令。返回 None 表示退出。"""
        return None


class ConsoleRenderer(Renderer):
    """终端渲染器（Rich 版）：用 Panel / JSON / Markdown / Rule 对
    "思考 / 回答 / 工具 / 结论"做视觉分层。

    内部用 _phase 记住当前处于哪个流式阶段（reasoning / content / idle），
    只在阶段切换时打印小标题，避免逐 token 重复打标题。

    Rich 自动处理 Windows 终端兼容性和颜色降级（256 → 16 → 无色），
    不再需要手写 ANSI 转义序列。
    """

    def __init__(self) -> None:
        self._phase = "idle"  # "idle" | "reasoning" | "content"
        self._line_started = False
        # highlight=False: 关闭 Rich 对纯文本的自动高亮（数字/URL 等），
        # 避免流式输出时把部分 token 误判为可高亮对象。
        # RichJSON / Markdown 等 Renderable 有自己的高亮逻辑，不受此影响。
        self._console = Console(highlight=False)
        # 保护终端：prompt_permission / prompt_user 可能从不同线程被调用
        # （如并发工具的权限确认），锁保证一次只有一个交互占据终端。
        self._prompt_lock = threading.Lock()

    # ----- 流式阶段管理 -----

    def _start_phase(self, phase: str, title: str, title_style: str) -> None:
        """进入一个流式阶段：若是新阶段，先收尾上一个，再打标题。"""
        if self._phase == phase:
            return
        if self._phase in ("reasoning", "content"):
            self._console.print()  # 换行收尾上一个阶段的流式输出
        self._console.print()  # 空行分隔
        self._console.print(title, style=title_style)
        sys.stdout.flush()
        self._phase = phase
        self._line_started = False

    def _end_stream(self) -> None:
        """结束流式阶段（工具调用 / 最终回答前调用）。"""
        if self._phase in ("reasoning", "content"):
            self._console.print()  # 换行收尾
        self._phase = "idle"
        self._line_started = False

    def _stream_piece(self, piece: str, prefix_style: str, text_style: str) -> None:
        """流式逐块打印，自动在每行开头插入竖线引导符。"""
        lines = piece.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                self._console.print()
                self._line_started = False
            if line:
                if not self._line_started:
                    self._console.print("│ ", end="", style=prefix_style, markup=False)
                    self._line_started = True
                self._console.print(line, end="", style=text_style, markup=False)
        sys.stdout.flush()

    # ----- Renderer 接口实现 -----

    def on_reasoning_delta(self, piece: str) -> None:
        self._start_phase("reasoning", "💭 思考过程", "bold dim bright_black")
        self._stream_piece(piece, prefix_style="dim bright_black", text_style="dim")

    def on_content_delta(self, piece: str) -> None:
        self._start_phase("content", "🤖 模型响应", "bold dim white")
        self._stream_piece(piece, prefix_style="bold dim white", text_style="dim white")

    def on_tool_call(self, tool_call) -> None:
        self._end_stream()
        name = tool_call.name
        arguments = tool_call.arguments

        if arguments:
            body = json.dumps(arguments, ensure_ascii=False, indent=2)
            try:
                content = RichJSON(body)
            except Exception:
                content = Text(body, style="dim")
        else:
            content = Text("(无参数)", style="dim italic")

        self._console.print()
        self._console.print(
            Panel(
                content,
                title=f"[bold]🔧 {name}[/bold]",
                title_align="left",
                border_style="yellow",
                padding=(0, 1),
            )
        )

        if name == "execute_command":
            self._console.print(Rule("输出", style="dim"))

    def on_command_output(self, line: str) -> None:
        self._console.print(line, end="", style="dim", markup=False)
        sys.stdout.flush()

    def on_checkpoint_error(self, error: str) -> None:
        self._end_stream()
        self._console.print()
        self._console.print(
            Panel(
                Text(error, style="red"),
                title="[bold]⚠ checkpoint 保存失败[/bold]",
                border_style="red",
                padding=(0, 1),
            )
        )

    def on_tool_result(self, tool_result) -> None:
        self._end_stream()
        if hasattr(tool_result, "to_dict"):
            tool_result = tool_result.to_dict()

        if tool_result.get("ok"):
            data = tool_result.get("data")
            body = json.dumps(data, ensure_ascii=False, indent=2)
            try:
                content = RichJSON(body)
            except Exception:
                content = Text(body, style="dim")
            self._console.print(
                Panel(
                    content,
                    title="[bold]✅ 工具结果[/bold]",
                    title_align="left",
                    border_style="green",
                    padding=(0, 1),
                )
            )
        else:
            err_text = str(tool_result.get("err", "未知错误"))
            self._console.print(
                Panel(
                    Text(err_text, style="red"),
                    title="[bold]❌ 工具失败[/bold]",
                    title_align="left",
                    border_style="red",
                    padding=(0, 1),
                )
            )

    def on_usage(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        context_limit: int | None,
    ) -> None:
        self._end_stream()
        inp = prompt_tokens if prompt_tokens is not None else "?"
        out = completion_tokens if completion_tokens is not None else "?"
        tot = total_tokens if total_tokens is not None else "?"

        # 上下文水位 = P+C:模型回复(C)已入队 messages,下次一定是输入的一部分,
        # 所以当前上下文的精确大小 = 本轮输入(P) + 本轮输出(C),两者都是服务端真值。
        if prompt_tokens is not None and completion_tokens is not None and context_limit:
            ctx_size = prompt_tokens + completion_tokens
            water = f"{ctx_size:,} / {context_limit:,} ({ctx_size / context_limit:.1%})"
        else:
            water = f"{inp} / ?"

        self._console.print()
        self._console.print(
            f"[dark_orange bold]tokens[/] "
            f"[dark_orange]输入 {inp} · 输出 {out} · 总计 {tot}[/]"
        )
        self._console.print(
            f"[dark_orange bold]context[/] "
            f"[dark_orange]水位 {water}[/]"
        )

    def on_context_compact(
        self,
        folded_count: int,
        prompt_tokens: int | None,
        context_limit: int | None,
        context_watermark: float,
    ) -> None:
        self._end_stream()

        if prompt_tokens is not None and context_limit:
            ctx_usage = f"{prompt_tokens:,} / {context_limit:,}"
            ctx_pct = f" ({prompt_tokens / context_limit:.1%})"
        else:
            ctx_usage = "? / ?"
            ctx_pct = ""
        watermark = f"{context_watermark:.0%}"

        if folded_count > 0:
            msg = f"已折叠 {folded_count} 条旧工具结果"
            style = "dark_orange"
        else:
            msg = "上下文已超水位,但暂无可折叠旧工具结果"
            style = "yellow"

        self._console.print()
        self._console.print(
            f"[{style} bold]context compact[/] "
            f"[{style}]{msg} · 水位 {ctx_usage}{ctx_pct} · 阈值 {watermark}[/]"
        )

    def on_final(self, answer) -> None:
        self._end_stream()
        self._console.print()

        # 字符串 → Markdown 渲染（LLM 回答通常是 Markdown 格式）
        # 非字符串（dict/list）→ JSON 语法高亮
        if isinstance(answer, str):
            content = Markdown(answer)
        else:
            body = json.dumps(answer, ensure_ascii=False, indent=2)
            try:
                content = RichJSON(body)
            except Exception:
                content = Text(body)

        self._console.print(
            Panel(
                content,
                title="[bold]💬 回答[/bold]",
                title_align="left",
                border_style="green",
                padding=(1, 2),
            )
        )

    # ── 双向交互 ──

    def prompt_permission(
        self,
        tool_name: str,
        subject: str,
        risk_flags: str,
        reason: str,
        offer_always: bool,
    ) -> str:
        with self._prompt_lock:
            self._end_stream()

            info = Text()
            info.append("工具: ", style="bold")
            info.append(f"{tool_name}\n")
            if subject:
                info.append("参数: ", style="bold")
                info.append(f"{subject}\n")
            info.append("风险: ", style="bold")
            info.append(f"{risk_flags}\n")
            info.append("说明: ", style="bold")
            info.append(reason)

            self._console.print()
            self._console.print(
                Panel(
                    info,
                    title="[bold]⚠️  需要权限确认[/bold]",
                    border_style="yellow",
                    padding=(0, 1),
                )
            )

            choices = "  [bold]y[/]=允许一次  [bold]n[/]=拒绝"
            if offer_always:
                choices += "  [bold]a[/]=本会话总是允许该工具"
            self._console.print(choices)

            prompt_text = HTML("  <b><ansiyellow>允许执行? </ansiyellow></b>")
            try:
                return prompt(prompt_text).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "n"

    def prompt_user(
        self,
        question: str,
        context: str = "",
        options: tuple[str, ...] = (),
    ) -> str | None:
        with self._prompt_lock:
            self._end_stream()

            body = Text()
            body.append(question, style="cyan")
            if context:
                body.append(f"\n{context}", style="dim")
            if options:
                body.append("\n")
                for idx, opt in enumerate(options, start=1):
                    body.append(f"\n  {idx}. {opt}", style="dim")

            self._console.print()
            self._console.print(
                Panel(
                    body,
                    title="[bold]❓ 需要你的回答[/bold]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )

            prompt_text = HTML("<b><ansicyan>你的回答 ❯ </ansicyan></b>")
            while True:
                try:
                    answer = prompt(prompt_text).strip()
                except (EOFError, KeyboardInterrupt):
                    self._console.print()
                    return None
                if answer:
                    return answer
                self._console.print("回答不能为空，请重新输入。", style="yellow")

    def prompt_main_input(self, prompt_session: Any = None) -> str | None:
        with self._prompt_lock:
            self._end_stream()
            self._console.print()

            prompt_text = HTML(
                "<b><ansicyan>╭─ 💬 你的指令 </ansicyan><ansibrightblack>(输入 /exit 退出)</ansibrightblack></b>\n"
                "<b><ansicyan>╰─❯ </ansicyan></b>"
            )

            try:
                if prompt_session is not None:
                    val = prompt_session.prompt(prompt_text).strip()
                else:
                    val = prompt(prompt_text).strip()
            except (EOFError, KeyboardInterrupt):
                self._console.print()
                return None

            if val and val not in ("/exit", "/quit"):
                # 按下回车后：抹掉两行输入提示符，原地替换为与最终答案规格一致的舒适卡片
                sys.stdout.write("\033[A\033[2K\033[A\033[2K\r")
                sys.stdout.flush()
                self._console.print(
                    Panel(
                        Markdown(val),
                        title="[bold]🧑 你的提问[/bold]",
                        title_align="left",
                        border_style="cyan",
                        padding=(1, 2),
                    )
                )
            return val


class SilentRenderer(Renderer):
    """静默渲染器：什么都不输出。用于测试或批量任务。"""

    def on_reasoning_delta(self, piece: str) -> None: ...
    def on_content_delta(self, piece: str) -> None: ...
    def on_tool_call(self, tool_call) -> None: ...
    def on_tool_result(self, tool_result) -> None: ...
    def on_final(self, answer) -> None: ...

