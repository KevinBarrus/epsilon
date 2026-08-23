"""拖选复制回调写入系统剪贴板的测试。"""

import asyncio

import pytest

from core import ui
from core.config import Settings
from core.status import create_status_info


class PlainClient:
    """返回纯文本回复的测试客户端。"""

    async def stream_response(self, messages, tools=(), thinking_level=None):
        yield ui.TextDelta("完成")

    async def stream_chat(self, messages):
        yield "摘要"


class FakeScreen:
    """捕获 on_copy 回调的假界面。"""

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
        self.copy_hint_provider = copy_hint_provider
        self.on_copy = on_copy
        self.hints: list[str] = []
        FakeScreen.instances.append(self)

    def add_entry(self, role: str, content: str, style: str = "") -> int:
        return 0

    def add_active_entry(self, role: str, content: str, style: str = "") -> int:
        return self.add_entry(role, content, style)

    def commit_entry(self, index: int) -> bool:
        return True

    def add_history_entries(self, entries) -> None:
        pass

    def append_to_entry(self, index: int, content: str) -> None:
        pass

    def set_entry_content(self, index: int, content: str) -> None:
        pass

    def set_tool_result(self, index: int, content: str) -> None:
        return None
    def set_entry_style(self, index: int, style: str) -> None:
        pass

    def set_status_message(self, message: str) -> None:
        pass


    def set_working(self, message: str | None, show_elapsed: bool = True) -> None:
        return None
    async def request_approval(self, definition, tool_call, allow_session=True):
        raise AssertionError("不应请求工具审批")

    async def request_skill_picker(self, items, checked):
        raise AssertionError("不应请求 skill 选择器")

    async def request_choice_picker(self, items, title, extra_options=None):
        raise AssertionError("不应请求选择器")

    async def request_text_input(self, title, is_password=False):
        raise AssertionError("不应请求文本输入")

    def invalidate(self) -> None:
        # 复制提示变化时记录当前提示文本
        if self.copy_hint_provider is not None:
            self.hints.append(self.copy_hint_provider())

    async def run_async(self) -> None:
        await self._on_submit("你好")


@pytest.mark.asyncio
async def test_on_copy_writes_clipboard_and_shows_hint(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试拖选复制回调把文本写入系统剪贴板并显示提示。"""

    written: list[str] = []
    monkeypatch.setattr(ui, "copy_text_to_clipboard", written.append)
    # 跳过 5 秒等待
    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(ui.asyncio, "sleep", fake_sleep)

    FakeScreen.instances = []
    monkeypatch.setattr(ui, "ChatScreen", FakeScreen)

    await ui.run_chat(
        PlainClient(),
        create_status_info("test-model", "n/a", tmp_path),
        Settings("https://example.com", "test", "key"),
        workspace=tmp_path,
    )

    screen = FakeScreen.instances[0]
    task = screen.on_copy("选中的文本内容")
    await asyncio.gather(task)

    assert written == ["选中的文本内容"]
    assert "Copied 7 chars to clipboard" in screen.hints
