"""终端主屏幕历史输出测试。"""

import pytest

from core.inline_history import InlineHistory


@pytest.mark.asyncio
async def test_inline_history_flushes_entries_in_order(capsys) -> None:
    """测试稳定历史按追加顺序输出，并在刷新后清空待处理队列。"""

    history = InlineHistory()
    history.extend(["第一条", "第二条"])

    await history.flush()

    assert capsys.readouterr().out == "第一条\n第二条\n"
    await history.flush()
    assert capsys.readouterr().out == ""
