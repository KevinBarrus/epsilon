"""提供命令补全单选列表组件。"""

from collections.abc import Callable

from prompt_toolkit.completion import Completion
from prompt_toolkit.formatted_text import AnyFormattedText, to_plain_text
from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.screen import Point
from wcwidth import wcswidth

# name 与 description 之间的最小间距（对齐 Pi PRIMARY_COLUMN_GAP）
_DESCRIPTION_GAP = 3


class CommandPicker:
    """渲染补全匹配项，管理选中与滚动。"""

    # 与 Codex 命令弹窗一致，最多同时展示 8 个稳定的单行选项
    _VISIBLE_ROWS = 8

    def __init__(
        self,
        completions: list[Completion],
        on_apply: Callable[[Completion], None] | None = None,
    ) -> None:
        """保存补全项并默认选中第一项，点击行时应用该补全。"""

        self._completions = list(completions)
        self._on_apply = on_apply
        self._cursor = 0
        self.window = Window(
            content=FormattedTextControl(
                self._render,
                focusable=True,
                show_cursor=False,
                get_cursor_position=self._cursor_position,
            ),
            height=Dimension(min=1, preferred=self._VISIBLE_ROWS, max=self._VISIBLE_ROWS),
            dont_extend_height=True,
            wrap_lines=False,
        )
        self.window.vertical_scroll = 0

    def move(self, offset: int) -> None:
        """移动光标，到达首尾后停留并保持选中项可见。"""

        if not self._completions:
            return
        self._cursor = max(0, min(len(self._completions) - 1, self._cursor + offset))
        self.window.content.reset()
        self._follow_cursor()

    def update_completions(self, completions: list[Completion]) -> None:
        """增量替换补全项，尽量保留当前选中项与滚动位置。"""

        new_texts = [completion.text for completion in completions]
        old_texts = [completion.text for completion in self._completions]
        if new_texts == old_texts:
            return
        current = self.selected.text if self.selected is not None else None
        self._completions = list(completions)
        if not new_texts:
            return
        for index, completion in enumerate(completions):
            if current is not None and completion.text == current:
                self._cursor = index
                break
        else:
            self._cursor = min(self._cursor, len(completions) - 1)
        self.window.content.reset()
        self._follow_cursor()

    @property
    def selected(self) -> Completion | None:
        """返回当前选中的补全项。"""

        if not self._completions:
            return None
        return self._completions[self._cursor]

    def _follow_cursor(self) -> None:
        """滚动窗口让当前光标项始终可见。"""

        if not self._completions:
            self.window.vertical_scroll = 0
            return

        max_scroll = max(0, len(self._completions) - self._VISIBLE_ROWS)
        scroll = self.window.vertical_scroll
        if self._cursor < scroll:
            scroll = self._cursor
        elif self._cursor >= scroll + self._VISIBLE_ROWS:
            scroll = self._cursor - self._VISIBLE_ROWS + 1
        self.window.vertical_scroll = max(0, min(max_scroll, scroll))

    def _cursor_position(self) -> Point | None:
        """返回选中项的实际渲染行，让 Window 保持该项可见。"""

        if not self._completions:
            return None
        return Point(x=0, y=self._cursor)

    def _click(self, index: int) -> None:
        """鼠标点击某行：选中该项并应用补全。"""

        self._cursor = index
        self.window.content.reset()
        self._follow_cursor()
        if self._on_apply is not None and self._completions:
            self._on_apply(self._completions[index])

    def _render(self) -> AnyFormattedText:
        """渲染固定单行命令项，保持选中状态不改变列表高度。"""

        names = [f"/{completion.text}" for completion in self._completions]
        metas = [
            to_plain_text(completion.display_meta)
            if completion.display_meta
            else ""
            for completion in self._completions
        ]
        name_width = max((wcswidth(name) for name in names), default=0)
        fragments: list[tuple[str, str, Callable]] = []
        for index, completion in enumerate(self._completions):
            prefix = "› " if index == self._cursor else "  "
            spacing = " " * max(1, name_width - wcswidth(names[index]) + _DESCRIPTION_GAP)
            handler = lambda event, i=index: self._click(i)
            if index == self._cursor:
                fragments.append(
                    ("class:approval-selected", f"{prefix}{names[index]}{spacing}{metas[index]}", handler)
                )
            else:
                fragments.append(("", f"{prefix}{names[index]}{spacing}", handler))
                if metas[index]:
                    fragments.append(
                        ("class:completion-description", metas[index], handler)
                    )
            fragments.append(("", "\n", handler))
        return fragments
