"""把已完成的对话内容输出到终端主屏幕回滚区。"""

import asyncio
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
        self._write_lock = asyncio.Lock()
        self._style = style or Style.from_dict({})
        self._output = output

    def add(self, text: AnyFormattedText) -> None:
        """追加一条非空历史内容。"""

        fragments = to_formatted_text(text)
        if fragments:
            self._pending.append(fragments)

    async def flush(self) -> None:
        """暂停活动界面并把待输出历史写入终端主屏幕。"""

        async with self._write_lock:
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

    async def replay(self, texts: Iterable[AnyFormattedText]) -> None:
        """清空终端缓冲区后，按当前宽度重新写入全部稳定历史。"""

        async with self._write_lock:
            entries = [to_formatted_text(text) for text in texts]
            self._pending = []
            async with in_terminal():
                output = self._output or get_app_session().output
                # CSI 3J 清除回滚区，后续操作清除可见区并将光标移到左上角
                output.write_raw("\x1b[3J")
                output.erase_screen()
                output.cursor_goto(0, 0)
                for fragments in entries:
                    print_formatted_text(output, fragments, self._style)
                    output.write("\n")
                output.flush()

    def extend(self, texts: Iterable[AnyFormattedText]) -> None:
        """按原顺序追加多条历史内容。"""

        for text in texts:
            self.add(text)
