"""工具调用三色背景在界面层的流转测试。"""

import pytest

from core import ui
from core.model import ToolCall
from core.status import create_status_info
from core.tools import ApprovalDecision
from core.config import Settings


class ToolTurnClient:
    """第一轮返回工具调用，第二轮返回文本。"""

    def __init__(self, tool_result_content: str = "文件内容") -> None:
        self.tool_result_content = tool_result_content
        self.requests: list[list[object]] = []

    async def stream_response(self, messages, tools=(), thinking_level=None):
        self.requests.append(list(messages))
        if len(self.requests) == 1:
            yield ui.ToolCallEvent(
                ToolCall("call-1", "read_file", {"path": "a.txt"})
            )
        else:
            yield ui.TextDelta("完成")

    async def stream_chat(self, messages):
        yield "摘要"


class FakeScreen:
    """记录工具条目样式流转的假界面。"""

    instances: list["FakeScreen"] = []

    def __init__(
        self,
        status,
        on_submit,
        command_names=None,
        model_name_provider=None,
        balance_text_provider=None,
        provider_name_provider=None,
        thinking_level_provider=None,
        info_line_provider=None,
        copy_hint_provider=None,
        on_copy=None,
        startup_info_provider=None,
    ) -> None:
        self._on_submit = on_submit
        self.application = self
        self.entries: list[tuple[str, str]] = []
        self.styles: list[str] = []
        self.initial_styles: list[str] = []
        self.style_updates: list[tuple[int, str]] = []
        self.committed_indices: list[int] = []
        FakeScreen.instances.append(self)

    def add_entry(self, role: str, content: str, style: str = "") -> int:
        self.entries.append((role, content))
        self.styles.append(style)
        self.initial_styles.append(style)
        return len(self.entries) - 1

    def add_active_entry(self, role: str, content: str, style: str = "") -> int:
        return self.add_entry(role, content, style)

    def commit_entry(self, index: int) -> bool:
        if index in self.committed_indices:
            return False
        self.committed_indices.append(index)
        return True

    def set_entry_style(self, index: int, style: str) -> None:
        self.styles[index] = style
        self.style_updates.append((index, style))

    def add_history_entries(self, entries) -> None:
        pass

    def append_to_entry(self, index: int, content: str) -> None:
        role, current = self.entries[index]
        self.entries[index] = (role, current + content)

    def set_entry_content(self, index: int, content: str) -> None:
        role, _ = self.entries[index]
        self.entries[index] = (role, content)

    def set_tool_result(self, index: int, content: str) -> None:
        return None
    def set_status_message(self, message: str) -> None:
        pass


    def set_working(self, message: str | None, show_elapsed: bool = True) -> None:
        return None
    async def request_approval(self, definition, tool_call, allow_session=True):
        return ApprovalDecision.ALLOW_ONCE

    async def request_skill_picker(self, items, checked):
        raise AssertionError("不应请求 skill 选择器")

    async def request_choice_picker(self, items, title, extra_options=None):
        raise AssertionError("不应请求选择器")

    async def request_text_input(self, title, is_password=False):
        raise AssertionError("不应请求文本输入")

    async def run_async(self) -> None:
        await self._on_submit("读取文件")


@pytest.mark.asyncio
async def test_tool_entry_flows_pending_to_success(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试工具条目样式从待执行流转为成功。"""

    (tmp_path / "a.txt").write_text("文件内容", encoding="utf-8")
    FakeScreen.instances = []
    monkeypatch.setattr(ui, "ChatScreen", FakeScreen)

    await ui.run_chat(
        ToolTurnClient(),
        create_status_info("test-model", "暂不可查询", tmp_path),
        Settings("https://example.com", "test", "key"),
        workspace=tmp_path,
    )

    screen = FakeScreen.instances[0]

    # 初始为待执行样式，执行后同一条目流转为成功
    assert "class:tool-pending" in screen.initial_styles
    assert screen.style_updates[-1][1] == "class:tool-success"
    assert screen.committed_indices == [1, 2, 3]


@pytest.mark.asyncio
async def test_tool_entry_flows_pending_to_error(tmp_path, monkeypatch) -> None:
    """测试工具执行失败时样式流转为错误。"""

    # 目标文件不存在，read_file 执行失败
    FakeScreen.instances = []
    monkeypatch.setattr(ui, "ChatScreen", FakeScreen)

    await ui.run_chat(
        ToolTurnClient(),
        create_status_info("test-model", "暂不可查询", tmp_path),
        Settings("https://example.com", "test", "key"),
        workspace=tmp_path,
    )

    screen = FakeScreen.instances[0]

    assert "class:tool-pending" in screen.initial_styles
    assert screen.style_updates[-1][1] == "class:tool-error"
    assert screen.committed_indices == [1, 2, 3]
