"""终端尺寸变化重排协调测试。"""

import asyncio

import pytest

from core.scrollback_reflow import ScrollbackReflow


@pytest.mark.asyncio
async def test_first_size_observation_does_not_reflow() -> None:
    """测试首次观察尺寸只初始化状态，不触发历史重建。"""

    calls: list[str] = []
    reflow = ScrollbackReflow(lambda: (80, 24), lambda: _record(calls), 0)

    assert reflow.observe(stream_active=False) is False
    await asyncio.sleep(0)

    assert calls == []


@pytest.mark.asyncio
async def test_resize_reflow_is_debounced() -> None:
    """测试连续尺寸变化只在最后一次变化后重排一次。"""

    size = [80, 24]
    calls: list[str] = []
    reflow = ScrollbackReflow(lambda: tuple(size), lambda: _record(calls), 0.01)

    reflow.observe(stream_active=False)
    size[0] = 90
    assert reflow.observe(stream_active=False) is True
    size[0] = 100
    assert reflow.observe(stream_active=False) is True
    await asyncio.sleep(0.02)

    assert calls == ["reflow"]


@pytest.mark.asyncio
async def test_stream_finish_requests_final_reflow() -> None:
    """测试流式期间尺寸变化后，提交条目会补一次最终重排。"""

    size = [80, 24]
    calls: list[str] = []
    reflow = ScrollbackReflow(lambda: tuple(size), lambda: _record(calls), 0)

    reflow.observe(stream_active=True)
    size[0] = 90
    reflow.observe(stream_active=True)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert reflow.stream_finished() is True
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert calls == ["reflow", "reflow"]


async def _record(calls: list[str]) -> None:
    """记录一次重排回调。"""

    calls.append("reflow")
