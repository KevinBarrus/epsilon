"""测试 slash command 在应用层的分发与 skill 注入。"""

from pathlib import Path

import pytest

import core.ui as ui
from core.status import create_status_info
from core.config import McpStdioSettings, Settings


class RecordingClient:
    """记录模型请求并返回固定回复。"""

    def __init__(self) -> None:
        self.requests: list[list[object]] = []

    async def stream_response(self, messages, tools=(), thinking_level=None):
        self.requests.append(list(messages))
        yield ui.TextDelta("完成")

    async def stream_chat(self, messages):
        yield "摘要"


@pytest.mark.asyncio
async def test_run_chat_start_skill_injects_active_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试 /start-skill 激活的 skill 会进入后续模型请求。"""

    skill_dir = tmp_path / ".epsilon" / "skills" / "git"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: git-commit\ndescription: 生成提交信息\n---\n规范提交正文",
        encoding="utf-8",
    )

    class FakeScreen:
        def __init__(self, status, on_submit, command_names=None, model_name_provider=None, balance_text_provider=None, provider_name_provider=None, thinking_level_provider=None, info_line_provider=None, copy_hint_provider=None, on_copy=None, startup_info_provider=None) -> None:
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
            pass

        def append_to_entry(self, index: int, content: str) -> None:
            pass

        def set_entry_content(self, index: int, content: str) -> None:
            pass

        def set_tool_result(self, index: int, content: str) -> None:
            return None
        def set_status_message(self, message: str) -> None:
            pass

        def set_working(self, message: str | None, show_elapsed: bool = True) -> None:
            return None

        async def request_approval(self, definition, tool_call, allow_session=True):
            raise AssertionError("不应请求工具审批")

        async def request_skill_picker(self, items, checked):
            return {("git-commit", "project")}

        async def run_async(self) -> None:
            await self._on_submit("/start-skill")
            await self._on_submit("继续")

    client = RecordingClient()
    monkeypatch.setattr(ui, "ChatScreen", FakeScreen)

    await ui.run_chat(
        client,
        create_status_info("test-model", "暂不可查询", tmp_path),
        workspace=tmp_path,
        settings=Settings("https://example.com", "test", "key"),
    )

    assert len(client.requests) == 1
    assert any(
        message.role == "system" and "规范提交正文" in message.content
        for message in client.requests[0]
    )


@pytest.mark.asyncio
async def test_run_chat_unknown_command_does_not_call_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试未知命令给出提示且不触发模型请求。"""

    class StrictClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            raise AssertionError("未知命令不应触发模型请求")

        async def stream_chat(self, messages):
            raise AssertionError("未知命令不应触发模型请求")

    class FakeScreen:
        instances: list["FakeScreen"] = []

        def __init__(self, status, on_submit, command_names=None, model_name_provider=None, balance_text_provider=None, provider_name_provider=None, thinking_level_provider=None, info_line_provider=None, copy_hint_provider=None, on_copy=None, startup_info_provider=None) -> None:
            self._on_submit = on_submit
            self.application = self
            self.entries: list[tuple[str, str]] = []
            FakeScreen.instances.append(self)

        def add_entry(self, role: str, content: str, style: str = "") -> int:
            self.entries.append((role, content))
            return len(self.entries) - 1

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
        def set_status_message(self, message: str) -> None:
            pass

        def set_working(self, message: str | None, show_elapsed: bool = True) -> None:
            return None

        async def request_approval(self, definition, tool_call, allow_session=True):
            raise AssertionError("不应请求工具审批")

        async def request_skill_picker(self, items, checked):
            raise AssertionError("不应请求 skill 选择器")

        async def run_async(self) -> None:
            await self._on_submit("/unknown-command")

    FakeScreen.instances = []
    monkeypatch.setattr(ui, "ChatScreen", FakeScreen)

    await ui.run_chat(
        StrictClient(),
        create_status_info("test-model", "暂不可查询", tmp_path),
        workspace=tmp_path,
        settings=Settings("https://example.com", "test", "key"),
    )

    assert any(
        role == "tool" and "Unknown command" in content
        for role, content in FakeScreen.instances[0].entries
    )
