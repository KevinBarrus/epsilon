from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from core import ui
from core.model import Message, ToolResult
from core.session import Session
from core.status import create_status_info
from core.tools import ApprovalDecision, ApprovalResult, ToolDefinition
from core.config import McpStdioSettings, Settings


class FakeApplication:
    """模拟 TUI 应用的启动"""

    async def run_async(self) -> None:
        """不启动真实终端"""


class FakeScreen:
    """记录恢复时的历史展示内容"""

    last: "FakeScreen | None" = None

    def __init__(self, status, on_submit, command_names=None, model_name_provider=None, balance_text_provider=None, provider_name_provider=None, thinking_level_provider=None, info_line_provider=None, copy_hint_provider=None, on_copy=None, startup_info_provider=None) -> None:
        """初始化假的界面对象"""

        self.entries: list[tuple[str, str]] = []
        self.history_batches: list[list[tuple[str, str]]] = []
        self.application = FakeApplication()
        FakeScreen.last = self

    def set_working(self, message, show_elapsed: bool = True):
        return None

    def add_entry(self, role: str, content: str, style: str = "") -> int:
        """记录一条展示消息"""

        self.entries.append((role, content))
        return len(self.entries) - 1

    def add_history_entries(self, entries: list[tuple[str, str]]) -> None:
        """记录恢复历史的批量追加调用。"""

        self.history_batches.append(entries)
        self.entries.extend(entries)

    async def request_approval(self, definition, tool_call, allow_session=True) -> ApprovalResult:
        """模拟界面审批回调"""

        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)


class EmptyClient:
    """不产生新模型回复的测试客户端"""

    async def stream_chat(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        """返回空的流式响应"""

        if False:
            yield ""


@pytest.mark.asyncio
async def test_run_chat_renders_restored_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试恢复会话时历史消息会立即展示"""

    session = Session(tmp_path)
    session.add_user_message("历史问题")
    session.add_assistant_message("历史回答")
    assert session.flush_persistence()
    monkeypatch.setattr(ui, "ChatScreen", FakeScreen)

    exit_info = await ui.run_chat(
        EmptyClient(),
        create_status_info("test", "暂不可查询", tmp_path),
        settings=Settings("https://example.com", "test", "key"),
        workspace=tmp_path,
        session_id=session.session_id,
    )

    assert FakeScreen.last is not None
    assert FakeScreen.last.entries == [
        ("user", "历史问题"),
        ("assistant", "历史回答"),
    ]
    assert FakeScreen.last.history_batches == [
        [("user", "历史问题"), ("assistant", "历史回答")]
    ]
    assert exit_info.session_id == session.session_id
    assert exit_info.usage_totals is None


@pytest.mark.asyncio
async def test_run_chat_registers_and_closes_mcp_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试 MCP 工具会进入模型定义，并在界面退出时关闭。"""

    captured: dict[str, object] = {}

    class FakeProvider:
        """提供一条可发现的远程只读工具。"""

        closed = False

        async def list_tools(self):
            return [
                ToolDefinition(
                    name="remote_echo",
                    description="远程回显",
                    parameters={"type": "object"},
                    source="mcp",
                    permission="read",
                    idempotent=True,
                    provider_id="demo",
                )
            ]

        async def call_tool(self, tool_call):
            return ToolResult(tool_call.call_id, "完成")

        async def close(self) -> None:
            self.closed = True

    class CapturingAgentLoop:
        """记录应用层注册后的模型工具定义。"""

        def __init__(self, client, tool_manager, max_tool_rounds=None) -> None:
            captured["tools"] = tool_manager.model_tools()

    provider = FakeProvider()
    monkeypatch.setattr(ui, "ChatScreen", FakeScreen)
    monkeypatch.setattr(ui, "AgentLoop", CapturingAgentLoop)

    await ui.run_chat(
        EmptyClient(),
        create_status_info("test", "暂不可查询", tmp_path),
        workspace=tmp_path,
        settings=Settings("https://example.com", "test", "key"),
        mcp_provider=provider,  # type: ignore[arg-type]
    )

    assert any(
        tool["function"]["name"] == "mcp_demo_remote_echo"
        for tool in captured["tools"]
    )
    assert provider.closed
