"""验证工具轮次耗尽后的应用层持久化"""

from pathlib import Path

import pytest

import core.ui as ui
from core.config import Settings
from core.model import ToolCall, ToolCallEvent
from core.session import Session
from core.status import create_status_info


class _ToolLimitScreen:
    """提供工具轮次测试所需的最小界面行为"""

    def __init__(self, status, on_submit, **kwargs) -> None:
        self._on_submit = on_submit
        self.application = self
        self.entries: list[tuple[str, str]] = []

    def add_entry(self, role: str, content: str, style: str = "") -> int:
        self.entries.append((role, content))
        return len(self.entries) - 1

    def add_active_entry(self, role: str, content: str, style: str = "") -> int:
        return self.add_entry(role, content, style)

    def commit_entry(self, index: int) -> bool:
        return True

    def add_history_entries(self, entries) -> None:
        self.entries.extend(entries)

    def append_to_entry(self, index: int, content: str) -> None:
        role, current = self.entries[index]
        self.entries[index] = (role, current + content)

    def set_entry_content(self, index: int, content: str) -> None:
        role, _ = self.entries[index]
        self.entries[index] = (role, content)

    def set_entry_style(self, index: int, style: str) -> None:
        return None

    def set_tool_result(self, index: int, content: str) -> None:
        return None

    def set_status_message(self, message: str) -> None:
        return None

    def set_working(self, message: str | None, show_elapsed: bool = True) -> None:
        return None

    def invalidate(self) -> None:
        return None

    async def request_approval(self, definition, tool_call, allow_session=True):
        raise AssertionError("本测试不应请求只读工具审批")

    async def run_async(self) -> None:
        await self._on_submit("读取说明")


class _ToolOnlyClient:
    """持续请求只读工具，用于触发工具轮次上限"""

    def __init__(self) -> None:
        self.requests = 0

    async def stream_response(self, messages, tools=(), thinking_level=None):
        self.requests += 1
        yield ToolCallEvent(
            ToolCall("call-1", "read_file", {"path": "README.md"})
        )


@pytest.mark.asyncio
async def test_run_chat_persists_tool_chain_when_round_limit_reached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试轮次耗尽后恢复会话仍能看到已执行工具和停止原因"""

    (tmp_path / "README.md").write_text("项目说明", encoding="utf-8")
    monkeypatch.setattr(ui, "ChatScreen", _ToolLimitScreen)
    client = _ToolOnlyClient()
    session_id = "00000000-0000-0000-0000-000000000019"

    await ui.run_chat(
        client,
        create_status_info("test", "unavailable", tmp_path),
        Settings("https://example.com", "test", "key"),
        workspace=tmp_path,
        session_id=session_id,
        max_tool_rounds=1,
    )

    restored = Session.restore(tmp_path, session_id)
    assert restored.get_messages() == [
        ui.Message(role="user", content="读取说明"),
        ui.Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall("call-1", "read_file", {"path": "README.md"}),),
        ),
        ui.Message(role="tool", content="项目说明", tool_call_id="call-1"),
        ui.Message(role="assistant", content=ui.TOOL_LIMIT_NOTICE),
    ]
    assert client.requests == 1
