"""构造 epsilon 的全屏终端界面。"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.formatted_text import (
    AnyFormattedText,
    StyleAndTextTuples,
    to_formatted_text,
    to_plain_text,
)
from prompt_toolkit.filters import has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    Container,
    VerticalAlign,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.filters import Condition
from prompt_toolkit.output import create_output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from wcwidth import wcswidth

from .clipboard import Osc52Clipboard
from .inline_history import InlineHistory
from .logo import DefaultLogoProvider, LogoProvider
from .markdown import render_markdown
from .status import format_cwd_for_footer


class StatusControl(UIControl):
    """两行状态栏控件：每行左右分区，右侧右对齐（对齐 Pi footer 整行渲染）。"""

    def __init__(self, get_rows) -> None:
        """get_rows 返回两行 (左侧文本, 右侧文本)。"""

        self._get_rows = get_rows

    def preferred_width(self, max_available_width: int) -> Dimension:
        """宽度跟随容器。"""

        return Dimension(min=1)

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_width: Callable[[str], int],
    ) -> Dimension:
        """固定两行高度。"""

        return Dimension.exact(2)

    def is_focusable(self) -> bool:
        """状态栏不可聚焦。"""

        return False

    def create_content(self, width: int, height: int) -> UIContent:
        """按容器宽度生成两行内容，右侧右对齐（对齐 Pi footer 整行渲染）。"""

        lines: list[StyleAndTextTuples] = []
        for left, right in self._get_rows()[:2]:
            gap = max(1, width - wcswidth(left) - (wcswidth(right) if right else 0))
            lines.append([("", left + " " * gap + right)])

        def get_line(lineno: int) -> StyleAndTextTuples:
            """返回指定行的片段。"""

            return lines[lineno] if lineno < len(lines) else []

        return UIContent(get_line=get_line, line_count=2)


from .status import StatusInfo
from .theme import create_ui_style
from .logo import DefaultLogoProvider, LogoProvider
from .model import ToolCall
from .command_picker import CommandPicker
from .skill_picker import SkillPicker
from .choice_picker import ChoicePicker
from .input_prompt import InputPrompt
from .tool_approval import ApprovalPrompt
from .tools.permissions import ApprovalResult
from .tools.types import ToolDefinition
from .ui_config import InputLayoutConfig


SubmitHandler = Callable[[str], Awaitable[None]]
ConversationRole = Literal["user", "assistant", "tool", "logo", "thinking", "working"]

# working 提示的转圈动画帧（对齐 Pi Loader）
_WORKING_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
# 工具输出折叠阈值（超过时显示省略提示，对齐 Pi）
_MAX_TOOL_LINES = 8
# 模型高频返回增量时，界面最多每秒重建 30 次 Markdown
_STREAM_RENDER_INTERVAL_SECONDS = 1 / 30


class SlashCommandCompleter(Completer):
    """输入 / 后按前缀匹配已注册的 slash command。"""

    def __init__(self, commands: list[tuple[str, str]]) -> None:
        """保存 (命令名, 描述) 列表。"""

        self._commands = list(commands)

    def get_completions(self, document, complete_event):
        """仅当输入以 / 开头时按 exact 优先、prefix 匹配生成补全。"""

        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        prefix = text[1:]
        exact: list[Completion] = []
        prefix_matches: list[Completion] = []
        for name, description in self._commands:
            if name == prefix:
                exact.append(
                    Completion(
                        name,
                        start_position=-len(prefix),
                        display_meta=description,
                    )
                )
            elif name.startswith(prefix):
                prefix_matches.append(
                    Completion(
                        name,
                        start_position=-len(prefix),
                        display_meta=description,
                    )
                )
        yield from exact
        yield from prefix_matches


def _render_assistant_content(
    content: str,
    *,
    streaming: bool = False,
) -> StyleAndTextTuples:
    """渲染助手回复：\x00 包裹的思考块用斜体灰，正文渲染 Markdown。"""

    if "\x00" not in content:
        return render_markdown(content, streaming=streaming)
    parts = content.split("\x00")
    fragments: StyleAndTextTuples = []
    if len(parts) > 1 and parts[1].strip():
        fragments.append(("class:md-thinking", parts[1] + "\n"))
    body = parts[2] if len(parts) > 2 else ""
    fragments.extend(render_markdown(body, streaming=streaming))
    return fragments


def _render_tool_diff(summary: str, diff_text: str) -> StyleAndTextTuples:
    """把工具结果摘要与 diff 文本渲染为红绿着色的片段。"""

    fragments: StyleAndTextTuples = [("", summary)]
    if not diff_text:
        return fragments
    fragments.append(("", "\n"))
    for line in diff_text.split("\n"):
        if line.startswith("+"):
            fragments.append(("class:tool-diff-add", line))
        elif line.startswith("-"):
            fragments.append(("class:tool-diff-del", line))
        else:
            fragments.append(("", line))
        fragments.append(("", "\n"))
    fragments.pop()
    return fragments


def _indent_markdown(
    fragments: StyleAndTextTuples, indent: str
) -> StyleAndTextTuples:
    """在 Markdown 片段的每行行首插入缩进，保留原有样式。"""

    result: StyleAndTextTuples = []
    at_line_start = True
    for style, text in fragments:
        if at_line_start and text:
            result.append((style, indent + text))
            at_line_start = False
        else:
            result.append((style, text))
        if "\n" in text:
            at_line_start = True
    return result


def _fragments_to_lines(
    fragments: StyleAndTextTuples,
) -> list[list[tuple[str, str]]]:
    """把带换行的样式片段拆成行列表，保留每段的样式。"""

    lines: list[list[tuple[str, str]]] = [[]]
    for style, text in fragments:
        parts = text.split("\n")
        for index, part in enumerate(parts):
            if index > 0:
                lines.append([])
            if part:
                lines[-1].append((style, part))
    return lines


@dataclass(frozen=True)
class ConversationEntry:
    """对话区中的一条展示内容。"""

    role: ConversationRole
    content: str
    control: FormattedTextControl
    style: str = ""
    committed: bool = True
    history_published: bool = False
    finalizing: bool = False


@dataclass(frozen=True)
class DraftState:
    """发送前输入框中的文字和光标位置。"""

    text: str
    cursor_position: int


class ChatScreen:
    """管理全屏界面的静态布局和可展示内容。"""

    def __init__(
        self,
        status: StatusInfo,
        style: Style | None = None,
        on_submit: SubmitHandler | None = None,
        logo_provider: LogoProvider | None = None,
        input_layout: InputLayoutConfig | None = None,
        command_names: list[tuple[str, str]] | None = None,
        model_name_provider: Callable[[], str] | None = None,
        balance_text_provider: Callable[[], str] | None = None,
        provider_name_provider: Callable[[], str] | None = None,
        thinking_level_provider: Callable[[], str] | None = None,
        info_line_provider: Callable[[], str] | None = None,
        copy_hint_provider: Callable[[], str] | None = None,
        startup_info_provider: Callable[[], list[list[tuple[str, str]]]] | None = None,
        inline_mode: bool = True,
    ) -> None:
        """创建对话区、输入区和两行式状态栏。"""

        self._status = status
        self._on_submit = on_submit
        self._logo_provider = logo_provider or DefaultLogoProvider()
        self._startup_info_provider = startup_info_provider
        self._input_layout = input_layout or InputLayoutConfig()
        self._model_name_provider = model_name_provider
        self._balance_text_provider = balance_text_provider
        self._provider_name_provider = provider_name_provider
        self._thinking_level_provider = thinking_level_provider
        self._info_line_provider = info_line_provider
        self._copy_hint_provider = copy_hint_provider
        self._inline_mode = inline_mode
        self._last_viewport_size: tuple[int, int] | None = None
        self._history_flush_task: asyncio.Task | None = None
        self._pending_stream_entries: set[int] = set()
        self._stream_render_task: asyncio.Task[None] | None = None
        self._finalize_tasks: set[asyncio.Task[None]] = set()
        self._request_active = False
        self._request_task: asyncio.Task[None] | None = None
        self._submitted_draft: DraftState | None = None
        self._approval_prompt: ApprovalPrompt | None = None
        self._skill_picker: SkillPicker | None = None
        self._choice_picker: ChoicePicker | None = None
        self._text_input: InputPrompt | None = None
        self._command_picker: CommandPicker | None = None
        self._status_message = ""
        self._working_message: str | None = None
        self._working_show_elapsed = True
        self._working_started = 0.0
        self._working_task: asyncio.Task | None = None
        self._working_entry_index: int | None = None
        self._expanded_entries: set[int] = set()
        self._last_tool_index: int | None = None
        self._conversation: list[ConversationEntry] = []
        self._input_history = InMemoryHistory()
        self._input_history_cursor: int | None = None
        self._restoring_input_history = False
        self._last_input_length = 0
        self.input_area = TextArea(
            prompt="",
            multiline=True,
            wrap_lines=True,
            scrollbar=False,
            height=Dimension(min=1, max=self._input_layout.max_lines),
            dont_extend_height=True,
            focus_on_click=True,
            style="class:input-area",
            get_line_prefix=self._get_input_line_prefix,
            history=self._input_history,
            completer=SlashCommandCompleter(command_names or []),
            complete_while_typing=True,
        )
        # 删除字符不会触发 complete_while_typing，需要手动重新启动补全
        self.input_area.buffer.on_text_changed += self._on_input_text_changed
        # 补全状态变化时同步底部区域的补全列表
        self.input_area.buffer.on_completions_changed += self._on_completions_changed
        self._conversation_content = HSplit(
            [],
            # 不添加顶部对齐的隐式填充行，消息内容只占实际需要的高度。
            align=VerticalAlign.JUSTIFY,
            padding=1,
        )
        # 输入区作为对话区内容末尾（对齐 Pi：随对话滚动，向上滚时下移出屏）
        self._build_input_container()
        self._input_tracker = self._input_container
        # 两行式状态栏（对齐 Pi footer）：整行渲染，行一左工作区右复制提示、行二左信息右模型
        self._status_control = StatusControl(self._status_rows)
        self._layout = Layout(self._create_layout())
        self._key_bindings = self._create_key_bindings()
        self.application = Application(
            layout=self._layout,
            key_bindings=self._key_bindings,
            style=style or create_ui_style(),
            full_screen=not inline_mode,
            mouse_support=False,
            cursor=CursorShape.BLINKING_BEAM,
            clipboard=Osc52Clipboard(),
            output=create_output(),
            min_redraw_interval=1 / 60,
        )
        # prompt-toolkit 显示光标时会关闭终端闪烁，渲染完成后恢复该模式
        self.application.after_render += self._enable_cursor_blink
        self.application.before_render += self._clear_stale_viewport_on_resize
        self._inline_history = InlineHistory(
            style=self.application.style,
            output=self.application.output,
        )
        # Logo 作为对话区第一条内容，随对话增长自然上移出屏幕
        if self._has_logo():
            self._conversation.append(
                ConversationEntry(
                    "logo",
                    "",
                    FormattedTextControl(self._render_logo, focusable=False),
                )
            )
            self._publish_entry(0)
        # 无论有无 Logo 都同步布局：输入区是对话内容的一部分
        self._sync_conversation()
        # 输入区在对话内容末尾，同步后焦点才可用
        self._layout.focus(self.input_area)

    def add_entry(self, role: ConversationRole, content: str, style: str = "") -> int:
        """向对话区追加一条展示内容，并返回它的索引。"""

        self._conversation.append(
            self._create_entry(role, content, style, committed=True)
        )
        self._publish_entry(len(self._conversation) - 1)
        self._sync_conversation()
        self.application.invalidate()
        return len(self._conversation) - 1

    def add_active_entry(
        self, role: ConversationRole, content: str, style: str = ""
    ) -> int:
        """追加一条仅在当前交互期间重绘的活动内容。"""

        self._conversation.append(
            self._create_entry(role, content, style, committed=False)
        )
        self._sync_conversation()
        self.application.invalidate()
        return len(self._conversation) - 1

    def commit_entry(self, index: int) -> bool:
        """提交活动条目，重复提交不产生新的历史输出。"""

        entry = self._conversation[index]
        if entry.committed or entry.finalizing:
            return False
        if entry.role == "assistant" and getattr(self.application, "_is_running", False):
            self._pending_stream_entries.discard(index)
            self._set_entry_content(index, entry.content, streaming=True)
            self._conversation[index] = replace(entry, finalizing=True)
            task = asyncio.create_task(
                self._finalize_assistant_entry(entry.control, entry.content)
            )
            self._finalize_tasks.add(task)
            task.add_done_callback(self._finalize_tasks.discard)
            return True
        return self._commit_entry_now(index)

    def _commit_entry_now(self, index: int) -> bool:
        """同步提交条目，供非助手条目和未运行的界面使用。"""

        entry = self._conversation[index]
        if entry.committed:
            return False
        self._conversation[index] = replace(entry, committed=True)
        self._set_entry_content(index, entry.content)
        self._publish_entry(index)
        self._sync_conversation()
        self.application.invalidate()
        return True

    def committed_entries(self) -> list[ConversationEntry]:
        """返回已完成且可写入终端历史的条目。"""

        return [entry for entry in self._conversation if entry.committed]

    def active_entries(self) -> list[ConversationEntry]:
        """返回仅供底部视口重绘的活动条目。"""

        return [entry for entry in self._conversation if not entry.committed]

    def add_history_entries(
        self,
        entries: list[tuple[ConversationRole, str]],
    ) -> None:
        """批量追加恢复历史，并只同步一次对话布局。"""

        if not entries:
            return
        self._conversation.extend(
            self._create_entry(role, content, committed=True)
            for role, content in entries
        )
        start = len(self._conversation) - len(entries)
        self._inline_history.extend(
            self._history_fragments(index)
            for index in range(start, len(self._conversation))
        )
        for index in range(start, len(self._conversation)):
            self._conversation[index] = replace(
                self._conversation[index], history_published=True
            )
        self._sync_conversation()
        self.application.invalidate()

    async def flush_history(self) -> None:
        """输出启动阶段积累的稳定历史。"""

        if self._finalize_tasks:
            await asyncio.gather(*self._finalize_tasks, return_exceptions=True)
        await self._inline_history.flush()

    def clear_conversation(self) -> None:
        """清空对话区展示内容（会话历史保留）。"""

        if self._inline_mode:
            # 主屏幕回滚区无法安全重写，只清空当前仍可重绘的活动条目
            self._conversation = [
                entry for entry in self._conversation if entry.committed
            ]
        else:
            self._conversation.clear()
        self._sync_conversation()
        self.application.invalidate()

    def _create_entry(
        self,
        role: ConversationRole,
        content: str,
        style: str = "",
        committed: bool = True,
    ) -> ConversationEntry:
        """创建保存独立文本控件的对话条目，助手消息渲染 Markdown。"""

        control_text = self._control_text(role, content, streaming=not committed)
        return ConversationEntry(
            role,
            content,
            FormattedTextControl(control_text, focusable=False),
            style,
            committed,
        )

    def _publish_entry(
        self,
        index: int,
        fragments: StyleAndTextTuples | None = None,
    ) -> None:
        """把稳定条目加入终端历史队列，并安排运行中的界面刷新。"""

        if not self._inline_mode:
            return
        entry = self._conversation[index]
        if not entry.committed or entry.history_published:
            return
        self._conversation[index] = replace(entry, history_published=True)
        self._inline_history.add(
            fragments if fragments is not None else self._history_fragments(index)
        )
        if getattr(self.application, "_is_running", False):
            if self._history_flush_task is None or self._history_flush_task.done():
                self._history_flush_task = asyncio.create_task(self._flush_history())

    async def _flush_history(self) -> None:
        """刷新稳定历史并清理刷新任务引用。"""

        try:
            await self._inline_history.flush()
        finally:
            self._history_flush_task = None

    def _enable_cursor_blink(self, _application: Application) -> None:
        """在每次绘制结束后恢复终端的光标闪烁模式。"""

        self.application.output.write_raw("\x1b[?12h")
        self.application.output.flush()

    def _clear_stale_viewport_on_resize(self, _application: Application) -> None:
        """尺寸变化时清除活动区遗留的一行旧输入边框。"""

        if not self._inline_mode:
            return
        size = self.application.output.get_size()
        viewport_size = (size.columns, size.rows)
        previous_size = self._last_viewport_size
        self._last_viewport_size = viewport_size
        if previous_size is None or previous_size == viewport_size:
            return
        # 框架已回到旧活动区顶部，额外向上一行覆盖遗留的上边框
        self.application.output.cursor_up(1)
        self.application.output.erase_down()
        self.application.output.flush()
        self.application.renderer.reset()

    def _history_fragments(
        self, index: int, expanded: bool = False
    ) -> StyleAndTextTuples:
        """把稳定条目转换为保留主题样式的终端历史片段。"""

        entry = self._conversation[index]
        if entry.role == "logo":
            return to_formatted_text(self._render_logo())
        if entry.role == "assistant":
            return _render_assistant_content(entry.content)
        if entry.role == "tool":
            return self._tool_entry_fragments(entry.content, expanded)
        if entry.role == "user":
            return self._user_history_fragments(entry.content)
        return [(entry.style, entry.content)]

    def _user_history_fragments(self, content: str) -> StyleAndTextTuples:
        """按终端当前宽度渲染用户消息，确保每个可见行都有灰色背景。"""

        # 最后一列不写满，避免终端自动换行把边框变成额外空白行
        columns = self.application.output.get_size().columns
        line_width = max(3, columns - 1)
        content_width = line_width - 2
        background = "class:conversation-user"
        lines: list[StyleAndTextTuples] = [[(background, " " * line_width)]]
        line: StyleAndTextTuples = [(background, " ")]
        used_width = 1

        def append_line() -> None:
            """补齐右侧留白并保存当前用户消息行。"""

            nonlocal line, used_width
            line.append((background, " " * (line_width - used_width)))
            lines.append(line)
            line = [(background, " ")]
            used_width = 1

        for style, text in render_markdown(content):
            styled = f"{background} {style}".strip()
            for character in text:
                if character == "\n":
                    append_line()
                    continue
                character_width = max(1, wcswidth(character))
                if used_width > 1 and used_width - 1 + character_width > content_width:
                    append_line()
                line.append((styled, character))
                used_width += character_width
        append_line()
        lines.append([(background, " " * line_width)])

        fragments: StyleAndTextTuples = []
        for line_index, row in enumerate(lines):
            if line_index:
                fragments.append(("", "\n"))
            fragments.extend(row)
        return fragments

    @staticmethod
    def _control_text(
        role: ConversationRole,
        content: str,
        *,
        streaming: bool = False,
    ) -> object:
        """按角色生成控件文本：用户消息渲染 Markdown 并留白，助手消息渲染思考块 + Markdown。"""

        if role == "assistant":
            return _render_assistant_content(content, streaming=streaming)
        if role == "user":
            fragments = render_markdown(content)
            return [("", "\n"), *_indent_markdown(fragments, " "), ("", "\n")]
        return content

    def append_to_entry(self, index: int, content: str) -> None:
        """向指定的对话条目追加流式文本。"""

        entry = self._conversation[index]
        self._conversation[index] = replace(entry, content=entry.content + content)
        if (
            entry.role == "assistant"
            and not entry.committed
            and not entry.finalizing
            and getattr(self.application, "_is_running", False)
        ):
            self._pending_stream_entries.add(index)
            if self._stream_render_task is None or self._stream_render_task.done():
                self._stream_render_task = asyncio.create_task(
                    self._render_stream_entries()
                )
        else:
            self._set_entry_content(index, self._conversation[index].content)
        self.application.invalidate()

    async def _render_stream_entries(self) -> None:
        """合并短时间内的模型增量，只重建一次活动回复。"""

        try:
            await asyncio.sleep(_STREAM_RENDER_INTERVAL_SECONDS)
            pending = self._pending_stream_entries
            self._pending_stream_entries = set()
            for index in pending:
                if index >= len(self._conversation):
                    continue
                entry = self._conversation[index]
                if entry.role == "assistant" and not entry.committed and not entry.finalizing:
                    self._set_entry_content(index, entry.content, streaming=True)
            self.application.invalidate()
        finally:
            self._stream_render_task = None

    async def _finalize_assistant_entry(
        self,
        control: FormattedTextControl,
        content: str,
    ) -> None:
        """在后台完成语法高亮，再提交已结束的助手回复。"""

        try:
            fragments = await asyncio.to_thread(_render_assistant_content, content)
        except Exception:
            fragments = _render_assistant_content(content, streaming=True)
        index = next(
            (
                current_index
                for current_index, entry in enumerate(self._conversation)
                if entry.control is control
            ),
            None,
        )
        if index is None:
            return
        entry = self._conversation[index]
        if not entry.finalizing or entry.content != content:
            return
        self._conversation[index] = replace(entry, committed=True, finalizing=False)
        entry.control.text = fragments
        entry.control.reset()
        self._publish_entry(index, fragments)
        self._sync_conversation()
        self.application.invalidate()

    def set_entry_content(self, index: int, content: str) -> None:
        """替换指定对话条目的展示内容。"""

        self._set_entry_content(index, content)
        self.application.invalidate()

    def set_entry_style(self, index: int, style: str) -> None:
        """替换指定对话条目的展示样式（如工具调用三色）。"""

        entry = self._conversation[index]
        if entry.style == style:
            return
        self._conversation[index] = replace(entry, style=style)
        self._sync_conversation()
        self.application.invalidate()

    def set_tool_result(self, index: int, content: str) -> None:
        """设置工具结果条目：按展开状态折叠 + diff 红绿渲染。"""

        entry = self._conversation[index]
        expanded = index in self._expanded_entries
        entry.control.text = self._tool_entry_fragments(content, expanded)
        entry.control.reset()
        self._conversation[index] = replace(entry, content=content)
        self._last_tool_index = index
        self.application.invalidate()

    def toggle_tool_expansion(self) -> None:
        """切换最近一个工具结果条目的展开/折叠状态。"""

        if self._last_tool_index is None:
            return
        index = self._last_tool_index
        if index in self._expanded_entries:
            self._expanded_entries.discard(index)
        else:
            self._expanded_entries.add(index)
        entry = self._conversation[index]
        entry.control.text = self._tool_entry_fragments(
            entry.content, index in self._expanded_entries
        )
        entry.control.reset()
        if self._inline_mode and index in self._expanded_entries:
            self._inline_history.add(self._history_fragments(index, expanded=True))
            if getattr(self.application, "_is_running", False):
                if self._history_flush_task is None or self._history_flush_task.done():
                    self._history_flush_task = asyncio.create_task(self._flush_history())
        self.application.invalidate()

    def _tool_entry_fragments(self, content: str, expanded: bool):
        """折叠超长工具输出，未折叠时按 diff 行渲染红绿。"""

        if not expanded:
            lines = content.split("\n")
            if len(lines) > _MAX_TOOL_LINES:
                visible = lines[:_MAX_TOOL_LINES]
                hidden = len(lines) - _MAX_TOOL_LINES
                return [
                    (
                        "",
                        "\n".join(visible)
                        + f"\n... {hidden} more lines (ctrl+o to expand)",
                    )
                ]
        summary, _, diff = content.partition("\n")
        return _render_tool_diff(summary, diff)

    def _set_entry_content(
        self,
        index: int,
        content: str,
        *,
        streaming: bool | None = None,
    ) -> None:
        """更新已有条目的内容和控件，不重建整个对话布局。"""

        entry = self._conversation[index]
        entry.control.text = self._control_text(
            entry.role,
            content,
            streaming=not entry.committed if streaming is None else streaming,
        )
        entry.control.reset()
        self._conversation[index] = replace(entry, content=content)

    def set_request_active(self, active: bool) -> None:
        """更新请求状态，避免模型响应期间重复提交。"""

        self._request_active = active

    def copy_input_selection(self) -> None:
        """复制输入框中选中的文本，没有选区时不执行操作。"""

        buffer = self.input_area.buffer
        if buffer.selection_state is None:
            return
        self.application.clipboard.set_data(buffer.copy_selection())

    def paste_to_input(self) -> None:
        """将剪贴板内容粘贴到输入框当前位置。"""

        self.input_area.buffer.paste_clipboard_data(
            self.application.clipboard.get_data()
        )

    def _create_layout(self) -> HSplit:
        """创建对话区、输入区和状态栏的垂直布局。"""

        self._conversation_container = ConditionalContainer(
            self._conversation_content,
            filter=Condition(lambda: True),
        )
        # 两行式状态栏（对齐 Pi footer）：整行渲染，无背景，右侧右对齐
        self._status_window = Window(
            content=self._status_control,
            height=2,
            style="class:status-bar",
        )
        # 行内模式把输入框固定在活动区底部，全屏兼容路径继续随对话滚动
        bottom_container = HSplit(self._default_bottom_children())
        self._bottom_container = bottom_container
        # 输入区已作为对话内容末尾（_build_input_container），根布局只含对话与底部
        return HSplit(
            [
                self._conversation_container,
                bottom_container,
            ],
            # 不使用 TOP，避免 prompt_toolkit 自动追加一个无样式的
            # 填充窗口；剩余空间应当只交给有消息时的对话视口。
            align=VerticalAlign.JUSTIFY,
        )

    def _default_bottom_children(self) -> list[Container]:
        """返回没有审批或选择器时的底部活动组件。"""

        if self._inline_mode:
            return [self._input_tracker, self._status_window]
        return [self._status_window]

    def _set_bottom_prompt(self, prompt: Container) -> None:
        """显示底部交互组件，并在行内模式保留状态栏。"""

        children = [prompt]
        if self._inline_mode:
            children.append(self._status_window)
        self._bottom_container.children = children

    def _build_input_container(self) -> None:
        """构建输入区（上下边框 + 输入框），作为对话区内容末尾。"""

        # 输入区上下各一条水平线框住（对齐 Pi DynamicBorder），不使用灰色背景；
        # 上边界在输入超行时左侧显示 ↑ n more 提示
        top_border = Window(
            content=FormattedTextControl(self._render_input_top_border),
            height=1,
            style="class:input-border",
            width=Dimension(weight=1),
        )
        border_line = Window(
            content=FormattedTextControl("─" * 4096),
            height=1,
            style="class:input-border",
            width=Dimension(weight=1),
        )
        self._input_container = HSplit(
            [
                top_border,
                self.input_area,
                border_line,
            ],
            # 输入区必须只占水平线和文字实际需要的高度。
            align=VerticalAlign.JUSTIFY,
            width=Dimension(weight=1),
        )
        # TextArea 自身已经限制了最大高度，直接放入对话内容 HSplit，
        # 避免用 Window 错误地包裹 Container，导致焦点控件无法被找到。
        self._input_window = self.input_area.window

    async def request_approval(
        self,
        definition: ToolDefinition,
        tool_call: ToolCall,
        allow_session: bool = True,
    ) -> ApprovalResult:
        """在输入区下方显示审批选项并等待用户选择。"""

        prompt = ApprovalPrompt(definition, tool_call, allow_session)
        self._approval_prompt = prompt
        self._set_bottom_prompt(prompt.window)
        self._layout.focus(prompt.window)
        self.application.invalidate()
        try:
            return await prompt.wait()
        finally:
            self._bottom_container.children = self._default_bottom_children()
            self._approval_prompt = None
            self._layout.focus(self.input_area)
            self.application.invalidate()

    async def request_skill_picker(
        self,
        items: list[tuple[str, str, str]],
        checked: set[tuple[str, str]],
    ) -> set[tuple[str, str]] | None:
        """在输入区下方显示 skill 勾选列表并等待用户选择。"""

        picker = SkillPicker(items, checked, on_interact=self.application.invalidate)
        self._skill_picker = picker
        self._set_bottom_prompt(picker.window)
        self._layout.focus(picker.window)
        self.application.invalidate()
        try:
            return await picker.wait()
        finally:
            self._bottom_container.children = self._default_bottom_children()
            self._skill_picker = None
            self._layout.focus(self.input_area)
            self.application.invalidate()

    async def request_choice_picker(
        self,
        items: list[str],
        title: str,
        extra_options: list[str] | None = None,
    ) -> str | None:
        """在输入区下方显示单选列表并等待用户选择。"""

        picker = ChoicePicker(items, title, extra_options, on_interact=self.application.invalidate)
        self._choice_picker = picker
        self._set_bottom_prompt(picker.window)
        self._layout.focus(picker.window)
        self.application.invalidate()
        try:
            return await picker.wait()
        finally:
            self._bottom_container.children = self._default_bottom_children()
            self._choice_picker = None
            self._layout.focus(self.input_area)
            self.application.invalidate()

    async def request_text_input(
        self,
        title: str,
        is_password: bool = False,
    ) -> str | None:
        """在输入区下方显示单行文本输入框并等待用户输入。"""

        prompt = InputPrompt(title, is_password)
        self._text_input = prompt
        self._set_bottom_prompt(prompt.window)
        self._layout.focus(prompt.input_area)
        self.application.invalidate()
        try:
            return await prompt.wait()
        finally:
            self._bottom_container.children = self._default_bottom_children()
            self._text_input = None
            self._layout.focus(self.input_area)
            self.application.invalidate()

    def _has_logo(self) -> bool:
        """判断 Logo 是否有实际内容，空 Logo 不占用布局空间。"""

        return bool(to_plain_text(self._logo_provider.render()).strip())

    def _get_input_line_prefix(self, lineno: int, wrap_count: int):
        """输入文字紧贴最左，不保留前缀空白（对齐 Pi 输入框）。"""

        return []

    def _render_input_top_border(self) -> str:
        """渲染输入框上边界：超行时左侧显示 ↑ n more 提示。"""

        hidden = self._hidden_input_lines()
        if hidden <= 0:
            return "─" * 4096
        return f"─── ↑ {hidden} more " + "─" * 4096

    def _hidden_input_lines(self) -> int:
        """计算输入框内容被隐藏的行数（超出最大行数后顶部滚出的行）。"""

        info = self.input_area.window.render_info
        if info is None:
            return 0
        return max(0, info.ui_content.line_count - info.window_height)

    def _create_key_bindings(self) -> KeyBindings:
        """创建提交、换行和退出快捷键。"""

        key_bindings = KeyBindings()
        input_focused = has_focus(self.input_area.buffer)
        approval_active = Condition(lambda: self._approval_prompt is not None)
        skill_picker_active = Condition(lambda: self._skill_picker is not None)
        choice_active = Condition(lambda: self._choice_picker is not None)
        text_input_active = Condition(lambda: self._text_input is not None)
        command_picker_active = Condition(lambda: self._command_picker is not None)
        embedded_active = (
            approval_active | skill_picker_active | choice_active | text_input_active
        )

        @key_bindings.add("up", filter=approval_active)
        def move_approval_up(event) -> None:
            """向上移动审批选项。"""

            if self._approval_prompt is not None:
                self._approval_prompt.move(-1)
                self.application.invalidate()

        @key_bindings.add("down", filter=approval_active)
        def move_approval_down(event) -> None:
            """向下移动审批选项。"""

            if self._approval_prompt is not None:
                self._approval_prompt.move(1)
                self.application.invalidate()

        @key_bindings.add("enter", filter=approval_active, eager=True)
        def confirm_approval(event) -> None:
            """确认当前审批选项。"""

            if self._approval_prompt is not None:
                self._approval_prompt.confirm()

        @key_bindings.add("escape", filter=approval_active)
        def reject_approval(event) -> None:
            """取消审批并拒绝工具调用。"""

            if self._approval_prompt is not None:
                self._approval_prompt.reject()

        @key_bindings.add("up", filter=skill_picker_active)
        def move_skill_up(event) -> None:
            """向上移动 skill 选择。"""

            if self._skill_picker is not None:
                self._skill_picker.move(-1)
                self.application.invalidate()

        @key_bindings.add("down", filter=skill_picker_active)
        def move_skill_down(event) -> None:
            """向下移动 skill 选择。"""

            if self._skill_picker is not None:
                self._skill_picker.move(1)
                self.application.invalidate()

        @key_bindings.add("space", filter=skill_picker_active)
        def toggle_skill(event) -> None:
            """切换当前 skill 的勾选状态。"""

            if self._skill_picker is not None:
                self._skill_picker.toggle()
                self.application.invalidate()

        @key_bindings.add("enter", filter=skill_picker_active, eager=True)
        def confirm_skill(event) -> None:
            """确认当前勾选的 skill 集合。"""

            if self._skill_picker is not None:
                self._skill_picker.confirm()

        @key_bindings.add("escape", filter=skill_picker_active)
        def cancel_skill(event) -> None:
            """取消 skill 选择。"""

            if self._skill_picker is not None:
                self._skill_picker.cancel()

        @key_bindings.add("up", filter=choice_active)
        def move_choice_up(event) -> None:
            """向上移动单选光标。"""

            if self._choice_picker is not None:
                self._choice_picker.move(-1)
                self.application.invalidate()

        @key_bindings.add("down", filter=choice_active)
        def move_choice_down(event) -> None:
            """向下移动单选光标。"""

            if self._choice_picker is not None:
                self._choice_picker.move(1)
                self.application.invalidate()

        @key_bindings.add("enter", filter=choice_active, eager=True)
        def confirm_choice(event) -> None:
            """确认当前单选项目。"""

            if self._choice_picker is not None:
                self._choice_picker.confirm()

        @key_bindings.add("escape", filter=choice_active)
        def cancel_choice(event) -> None:
            """取消单选选择。"""

            if self._choice_picker is not None:
                self._choice_picker.cancel()

        @key_bindings.add("enter", filter=text_input_active, eager=True)
        def confirm_text_input(event) -> None:
            """确认文本输入。"""

            if self._text_input is not None:
                self._text_input.confirm()

        @key_bindings.add("escape", filter=text_input_active)
        def cancel_text_input(event) -> None:
            """取消文本输入。"""

            if self._text_input is not None:
                self._text_input.cancel()

        @key_bindings.add(
            "enter",
            filter=~embedded_active,
            eager=True,
        )
        def submit(event) -> None:
            """提交输入框中的内容，补全列表打开时先应用选中项。"""

            buffer = self.input_area.buffer
            completion = self._selected_completion(buffer)
            if completion is not None:
                buffer.apply_completion(completion)
            prompt = self.input_area.text.strip()
            if not prompt or self._request_active or self._on_submit is None:
                return
            self._submitted_draft = DraftState(
                text=self.input_area.text,
                cursor_position=buffer.cursor_position,
            )
            self._input_history.append_string(self.input_area.text)
            self._input_history_cursor = None
            self.input_area.text = ""
            self._request_task = event.app.create_background_task(
                self._submit(prompt)
            )

        @key_bindings.add("up", filter=command_picker_active, eager=True)
        def move_command_up(event) -> None:
            """向上移动补全列表选中项。"""

            if self._command_picker is not None:
                self._command_picker.move(-1)
                self.application.invalidate()

        @key_bindings.add("down", filter=command_picker_active, eager=True)
        def move_command_down(event) -> None:
            """向下移动补全列表选中项。"""

            if self._command_picker is not None:
                self._command_picker.move(1)
                self.application.invalidate()

        @key_bindings.add(
            "up",
            filter=input_focused & ~embedded_active & ~command_picker_active,
            eager=True,
        )
        def move_input_up(event) -> None:
            """按 Codex 规则处理输入框上移与历史恢复。"""

            buffer = event.current_buffer
            if self._input_history_cursor is not None or not buffer.text:
                self._navigate_input_history(-1)
            else:
                buffer.cursor_up()
            self.application.invalidate()

        @key_bindings.add(
            "down",
            filter=input_focused & ~embedded_active & ~command_picker_active,
            eager=True,
        )
        def move_input_down(event) -> None:
            """按 Codex 规则处理输入框下移与历史恢复。"""

            buffer = event.current_buffer
            if self._input_history_cursor is not None:
                self._navigate_input_history(1)
            else:
                buffer.cursor_down()
            self.application.invalidate()

        @key_bindings.add(
            "tab",
            filter=command_picker_active & input_focused & ~embedded_active,
            eager=True,
        )
        def apply_completion(event) -> None:
            """Tab 应用当前补全但不提交（对齐 Codex）。"""

            buffer = self.input_area.buffer
            completion = self._selected_completion(buffer)
            if completion is not None:
                buffer.apply_completion(completion)
                self.application.invalidate()

        @key_bindings.add("c-o", filter=input_focused)
        def toggle_tool_expand(event) -> None:
            """展开/折叠最近一个工具结果（对齐 Pi app.tools.expand）。"""

            self.toggle_tool_expansion()

        @key_bindings.add("c-j", filter=input_focused)
        def insert_newline(event) -> None:
            """使用兼容快捷键在输入框中插入换行。"""

            # 普通终端通常无法区分 Ctrl+Enter 和 Enter，因此使用 Ctrl+J 作为可靠备用键。
            event.current_buffer.insert_text("\n")

        @key_bindings.add("c-d")
        def exit_application(event) -> None:
            """使用 Ctrl+D 退出全屏界面。"""

            event.app.exit()

        @key_bindings.add("c-c", filter=input_focused)
        def copy_selection(event) -> None:
            """复制输入框中的选中文本。"""

            self.copy_input_selection()

        @key_bindings.add("c-v", filter=input_focused)
        @key_bindings.add("s-insert", filter=input_focused)
        def paste_clipboard(event) -> None:
            """将剪贴板内容粘贴到输入框。"""

            self.paste_to_input()

        @key_bindings.add(
            "escape",
            filter=~embedded_active,
        )
        def cancel_request(event) -> None:
            """取消当前请求，输入恢复由请求任务负责。"""

            self.cancel_request()

        return key_bindings

    def cancel_request(self) -> None:
        """取消正在运行的模型请求。"""

        if self._request_task is not None and not self._request_task.done():
            self._request_task.cancel()

    def _selected_completion(self, buffer) -> object | None:
        """返回补全列表当前选中的补全项。"""

        if self._command_picker is not None:
            return self._command_picker.selected
        return None

    def _on_completions_changed(self, buffer) -> None:
        """补全状态变化时，在底部区域显示补全列表或恢复状态栏。

        注意：prompt_toolkit 补全加载期间每 0.3 秒触发一次本回调，
        因此列表存在时只做增量更新（保留选中项与滚动位置），不重建。
        """

        state = buffer.complete_state
        completions = list(state.completions) if state is not None else []
        if completions:
            if self._command_picker is None:
                self._command_picker = CommandPicker(
                    completions, on_apply=self._apply_clicked_completion
                )
                self._set_bottom_prompt(self._command_picker.window)
            else:
                self._command_picker.update_completions(completions)
        else:
            self._command_picker = None
            self._bottom_container.children = self._default_bottom_children()
        self.application.invalidate()

    def _apply_clicked_completion(self, completion: Completion) -> None:
        """应用鼠标点击选中的补全项（对齐 Tab 行为）。"""

        self.input_area.buffer.apply_completion(completion)
        self.application.invalidate()

    def _on_input_text_changed(self, buffer) -> None:
        """文本变化时维护命令补全：删字符后重新触发，不再以 / 开头时收起列表。

        注意：prompt_toolkit 的 _text_changed 清空 complete_state 时不会触发
        on_completions_changed，因此删除到非 / 前缀时要显式收起残留的补全列表。
        """

        text = buffer.text
        if text and not self._restoring_input_history:
            self._input_history_cursor = None
        if not text.startswith("/"):
            # 命令补全不再适用：清除残留列表并恢复状态栏
            if self._command_picker is not None:
                self._command_picker = None
                self._bottom_container.children = self._default_bottom_children()
                self.application.invalidate()
        elif len(text) < self._last_input_length and buffer.complete_state is None:
            buffer.start_completion()
        self._last_input_length = len(text)

    async def _submit(self, prompt: str) -> None:
        """标记请求状态并调用应用层提交处理器。"""

        if self._on_submit is None:
            return
        self._request_active = True
        try:
            await self._on_submit(prompt)
        except asyncio.CancelledError:
            self._restore_submitted_draft()
        finally:
            self._request_active = False
            self._request_task = None
            self._submitted_draft = None

    def _restore_submitted_draft(self) -> None:
        """恢复请求发送前的输入文本和光标位置。"""

        if self._submitted_draft is None:
            return

        draft = self._submitted_draft
        self.input_area.buffer.text = draft.text
        self.input_area.buffer.cursor_position = min(
            draft.cursor_position,
            len(draft.text),
        )
        self.application.invalidate()

    def _navigate_input_history(self, direction: int) -> None:
        """在当前会话历史中移动，并把光标放到恢复文本末尾。"""

        history = self._input_history.get_strings()
        if not history:
            return
        if direction < 0:
            if self._input_history_cursor is None:
                self._input_history_cursor = len(history) - 1
            else:
                self._input_history_cursor = max(0, self._input_history_cursor - 1)
        elif self._input_history_cursor is not None:
            if self._input_history_cursor >= len(history) - 1:
                self._input_history_cursor = None
                self._restoring_input_history = True
                try:
                    self.input_area.buffer.text = ""
                finally:
                    self._restoring_input_history = False
                return
            self._input_history_cursor += 1
        else:
            return

        text = history[self._input_history_cursor]
        self._restoring_input_history = True
        try:
            self.input_area.buffer.text = text
            self.input_area.buffer.cursor_position = len(text)
        finally:
            self._restoring_input_history = False

    def _render_entry(self, index: int) -> str:
        """返回对话条目的纯文本，不添加角色前缀。"""

        return self._conversation[index].content

    def _render_logo(self) -> AnyFormattedText:
        """渲染 Logo 与新建会话时的起始信息，整体按终端宽度居中。"""

        logo_lines = _fragments_to_lines(
            to_formatted_text(self._logo_provider.render())
        )
        lines = list(logo_lines)
        if self._startup_info_provider is not None:
            startup_lines = [
                list(info_line) for info_line in self._startup_info_provider()
            ]
            if startup_lines:
                # Logo 与前置信息、前置信息各行之间留出空行，增强启动页层次
                lines.append([])
                for index, info_line in enumerate(startup_lines):
                    if index:
                        lines.append([])
                    lines.append(info_line)
        width = self._terminal_width()
        fragments: list[tuple[str, str]] = []
        for line in lines:
            line_width = wcswidth("".join(text for _, text in line))
            indent = max(0, (width - line_width) // 2) if width else 0
            fragments.append(("", " " * indent))
            fragments.extend(line)
            fragments.append(("", "\n"))
        if fragments:
            fragments.pop()
        return fragments

    def _terminal_width(self) -> int:
        """返回终端当前列数，无法获取时返回 0。"""

        try:
            return self.application.output.get_size().columns
        except Exception:
            return 0

    def _sync_conversation(self) -> None:
        """将对话数据同步到带样式的可滚动视图。"""

        children: list[Container] = []
        visible_entries = (
            self._conversation
            if not self._inline_mode
            else [entry for entry in self._conversation if not entry.committed]
        )
        for index, entry in enumerate(visible_entries):
            if entry.role == "user":
                # 用户消息文本自带上下留白行，单窗口整体灰色背景（n 行内容展示 n+2 行全灰）
                children.append(
                    Window(
                        content=entry.control,
                        style="class:conversation-user",
                        wrap_lines=True,
                        dont_extend_height=True,
                    )
                )
            elif entry.role == "tool":
                children.append(
                    Window(
                        content=entry.control,
                        style=entry.style or "class:tool-activity",
                        wrap_lines=True,
                        dont_extend_height=True,
                    )
                )
            elif entry.role == "thinking":
                children.append(
                    Window(
                        content=entry.control,
                        style="class:md-thinking",
                        wrap_lines=True,
                        dont_extend_height=True,
                    )
                )
            elif entry.role == "working":
                children.append(
                    Window(
                        content=entry.control,
                        style="class:tool-activity",
                        wrap_lines=True,
                        dont_extend_height=True,
                    )
                )
            else:
                children.append(
                    Window(
                        content=entry.control,
                        wrap_lines=True,
                        dont_extend_height=True,
                    )
                )
        # 全屏兼容路径把输入区放入对话内容，行内模式由底部活动区承载
        if not self._inline_mode:
            children.append(self._input_tracker)
        self._conversation_content.children = children

    def _status_rows(self) -> list[tuple[str, str]]:
        """返回状态栏两行内容（左侧、右侧），供 StatusControl 整行右对齐渲染。"""

        cwd = format_cwd_for_footer(
            str(self._status.working_directory), str(Path.home())
        )
        left1 = cwd
        if self._status_message:
            left1 = f"{left1}   {self._status_message}"
        hint = (
            self._copy_hint_provider()
            if self._copy_hint_provider is not None
            else ""
        )
        info = (
            self._info_line_provider()
            if self._info_line_provider is not None
            else None
        )
        if info is None:
            balance = (
                self._balance_text_provider()
                if self._balance_text_provider is not None
                else self._status.balance
            )
            info = f"Balance: {balance}"
        model_name = (
            self._model_name_provider()
            if self._model_name_provider is not None
            else self._status.model_name
        )
        provider = (
            self._provider_name_provider()
            if self._provider_name_provider is not None
            else ""
        )
        thinking = (
            self._thinking_level_provider()
            if self._thinking_level_provider is not None
            else ""
        )
        parts: list[str] = []
        if provider:
            parts.append(f"({provider})")
        parts.append(model_name)
        if thinking:
            parts.append(f"· {thinking}")
        return [(left1, hint), (info, " ".join(parts))]

    def set_status_message(self, message: str) -> None:
        """更新状态栏中的运行时提示。"""

        self._status_message = message
        self.application.invalidate()

    def set_working(self, message: str | None, show_elapsed: bool = True) -> None:
        """设置对话区临时 working 条目（spinner + 消息，可选耗时）。"""

        self._remove_working_entry()
        self._working_message = message
        self._working_show_elapsed = show_elapsed
        self._working_started = time.monotonic() if message else 0.0
        if message:
            self._working_entry_index = self.add_active_entry(
                "working", self._working_text()
            )
        if message and self._working_task is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                self._working_task = loop.create_task(self._animate_working())
        elif message is None:
            self._working_task = None
        self.application.invalidate()

    def _remove_working_entry(self) -> None:
        """移除对话区中的临时 working 条目。"""

        if self._working_entry_index is None:
            return
        if self._working_entry_index < len(self._conversation):
            self._conversation.pop(self._working_entry_index)
            self._sync_conversation()
        self._working_entry_index = None

    def _working_text(self) -> str:
        """生成当前 working 提示文本，spinner 帧按时间轮转。"""

        if not self._working_message:
            return ""
        frame = _WORKING_FRAMES[
            int(time.monotonic() * 12.5) % len(_WORKING_FRAMES)
        ]
        text = f"{frame} {self._working_message}"
        if self._working_show_elapsed:
            elapsed = int(time.monotonic() - self._working_started)
            text = f"{text} {elapsed}s"
        return text

    async def _animate_working(self) -> None:
        """working 提示存在期间持续刷新界面（spinner 动画与秒数）。"""

        while self._working_message:
            if self._working_entry_index is not None:
                self._set_entry_content(
                    self._working_entry_index,
                    self._working_text(),
                )
            self.application.invalidate()
            await asyncio.sleep(0.08)
