"""把已完成的对话内容输出到终端主屏幕回滚区。"""

from collections.abc import Iterable

from prompt_toolkit.application.current import get_app_session
from prompt_toolkit.application.run_in_terminal import in_terminal
from prompt_toolkit.formatted_text import AnyFormattedText, to_formatted_text
from prompt_toolkit.output import Output
from prompt_toolkit.renderer import print_formatted_text
from prompt_toolkit.styles import BaseStyle, Style


class InlineHistory:
    """暂存并批量输出稳定历史，不参与活动界面渲染。"""

    def __init__(
        self,
        style: BaseStyle | None = None,
        output: Output | None = None,
    ) -> None:
        """创建尚未输出的历史队列。"""

        self._pending: list[AnyFormattedText] = []
        self._style = style or Style.from_dict({})
        self._output = output

    def add(self, text: AnyFormattedText) -> None:
        """追加一条非空历史内容。"""

        fragments = to_formatted_text(text)
        if fragments:
            self._pending.append(fragments)

    async def flush(self) -> None:
        """暂停活动界面并把待输出历史写入终端主屏幕。"""

        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        async with in_terminal():
            output = self._output or get_app_session().output
            for fragments in pending:
                print_formatted_text(output, fragments, self._style)
                output.write("\n")
            output.flush()

    def extend(self, texts: Iterable[AnyFormattedText]) -> None:
        """按原顺序追加多条历史内容。"""

        for text in texts:
            self.add(text)
