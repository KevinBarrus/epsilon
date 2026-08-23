"""把已完成的对话内容输出到终端主屏幕回滚区。"""

from collections.abc import Iterable

from prompt_toolkit.application.run_in_terminal import in_terminal


class InlineHistory:
    """暂存并批量输出稳定历史，不参与活动界面渲染。"""

    def __init__(self) -> None:
        """创建尚未输出的历史队列。"""

        self._pending: list[str] = []

    def add(self, text: str) -> None:
        """追加一条非空历史内容。"""

        if text:
            self._pending.append(text)

    async def flush(self) -> None:
        """暂停活动界面并把待输出历史写入终端主屏幕。"""

        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        async with in_terminal():
            for text in pending:
                print(text)

    def extend(self, texts: Iterable[str]) -> None:
        """按原顺序追加多条历史内容。"""

        for text in texts:
            self.add(text)
