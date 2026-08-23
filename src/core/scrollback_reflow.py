"""协调终端尺寸变化后的行内历史重排。"""

import asyncio
from collections.abc import Awaitable, Callable


TerminalSize = tuple[int, int]
SizeProvider = Callable[[], TerminalSize | None]
ReflowHandler = Callable[[], Awaitable[None]]


class ScrollbackReflow:
    """对终端尺寸变化进行尾部防抖，并在流式结束后补一次重排。"""

    def __init__(
        self,
        size_provider: SizeProvider,
        on_reflow: ReflowHandler,
        delay_seconds: float = 0.075,
    ) -> None:
        """保存尺寸来源、重排回调和防抖时长。"""

        self._size_provider = size_provider
        self._on_reflow = on_reflow
        self._delay_seconds = delay_seconds
        self._last_observed_size: TerminalSize | None = None
        self._last_reflow_size: TerminalSize | None = None
        self._task: asyncio.Task[None] | None = None
        self._stream_resize_pending = False

    def observe(self, stream_active: bool) -> bool:
        """记录一次绘制前的终端尺寸，变化时安排尾部重排。"""

        size = self._size_provider()
        if size is None:
            return False
        previous = self._last_observed_size
        self._last_observed_size = size
        if previous is None:
            self._last_reflow_size = size
            return False
        if previous == size:
            return False
        if stream_active:
            self._stream_resize_pending = True
        return self._schedule(self._delay_seconds)

    def stream_finished(self) -> bool:
        """在流式条目提交后补一次立即重排。"""

        if not self._stream_resize_pending:
            return False
        self._stream_resize_pending = False
        return self._schedule(0)

    def _schedule(self, delay_seconds: float) -> bool:
        """取消旧任务，只保留当前终端尺寸对应的重排。"""

        if self._task is not None and not self._task.done():
            self._task.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._task = loop.create_task(self._run_after_delay(delay_seconds))
        return True

    async def _run_after_delay(self, delay_seconds: float) -> None:
        """等待终端尺寸稳定后执行一次重排。"""

        try:
            if delay_seconds:
                await asyncio.sleep(delay_seconds)
            await self._on_reflow()
            self._last_reflow_size = self._last_observed_size
        finally:
            if self._task is asyncio.current_task():
                self._task = None
