"""新增 slash command（skills/mcp/compact/status/copy/clear/quit/export/diff）测试。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from core import ui
from core.commands import (
    CommandContext,
    clear_command_slash,
    compact_command_slash,
    copy_command_slash,
    diff_command_slash,
    export_command_slash,
    mcp_command_slash,
    quit_command_slash,
    skills_command_slash,
    status_command_slash,
)
from core.context import ContextBuildResult, ContextManager, ContextSummaryError
from core.model import Message
from core.session_store import CompactionRecord
from core.tools.types import ToolDefinition


class _Screen:
    """记录 add_entry 调用、选择器输入与退出请求的假界面。"""

    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []
        self.choices: list[str | None] = []
        self.exited = False
        self.application = SimpleNamespace(exit=lambda: setattr(self, "exited", True))

    def add_entry(self, role: str, content: str, style: str = "") -> int:
        self.entries.append((role, content))
        return len(self.entries) - 1

    def clear_conversation(self) -> None:
        self.entries.append(("tool", "cleared"))

    async def request_choice_picker(
        self, items, title, extra_options=None
    ) -> str | None:
        return self.choices.pop(0) if self.choices else None


def _context(**overrides) -> CommandContext:
    """构造最小可用命令上下文，缺省字段用桩对象填充。"""

    defaults = {
        "screen": _Screen(),
        "session": SimpleNamespace(
            session_id=None,
            mark_deleted=lambda: None,
            get_messages=lambda: [], get_compactions=lambda: [], add_compaction=lambda r: True
        ),
        "skill_manager": SimpleNamespace(
            list_skills=lambda: [], active_keys=lambda: set()
        ),
        "context_manager": SimpleNamespace(
            build_for_model_result=lambda *a, **k: ContextBuildResult([]),
            estimate_tokens=lambda messages: 100,
            context_window=lambda: 128000,
        ),
        "client_holder": SimpleNamespace(
            client=object(),
            settings=SimpleNamespace(
                model_name="deepseek-v4-pro",
                base_url="https://api.deepseek.com/",
            )
        ),
        "agent_loop": SimpleNamespace(thinking_level="high"),
        "project_dir": Path("."),
        "tool_manager": SimpleNamespace(list_definitions=lambda: []),
    }
    defaults.update(overrides)
    return CommandContext(**defaults)


def _skill(name: str, source: str, description: str = "") -> SimpleNamespace:
    return SimpleNamespace(name=name, description=description, source=source)


@pytest.mark.asyncio
async def test_skills_command_lists_active_state() -> None:
    """测试 /skills 展示激活与未激活的 skill。"""

    screen = _Screen()
    context = _context(
        screen=screen,
        skill_manager=SimpleNamespace(
            list_skills=lambda: [
                _skill("grill-me", "project", "提问练习"),
                _skill("teach", "global"),
            ],
            active_keys=lambda: {("grill-me", "project")},
        ),
    )

    await skills_command_slash.handler(context)

    content = screen.entries[0][1]
    assert "[on] grill-me (projects)" in content
    assert "[off]teach (global)" in content


@pytest.mark.asyncio
async def test_mcp_command_lists_only_mcp_tools() -> None:
    """测试 /mcp 只列出 MCP 来源的工具。"""

    screen = _Screen()
    context = _context(
        screen=screen,
        tool_manager=SimpleNamespace(
            list_definitions=lambda: [
                ToolDefinition("local_tool", "本地工具", {}, "local", "read", False),
                ToolDefinition("mcp_tool", "MCP 工具", {}, "mcp", "write", False),
            ]
        ),
    )

    await mcp_command_slash.handler(context)

    content = screen.entries[0][1]
    assert "mcp_tool" in content
    assert "local_tool" not in content


@pytest.mark.asyncio
async def test_compact_command_persists_compaction() -> None:
    """测试 /compact 生成并保存压缩记录。"""

    screen = _Screen()
    added: list[CompactionRecord] = []

    async def build(*args, **kwargs) -> ContextBuildResult:
        return ContextBuildResult(
            [],
            compaction=CompactionRecord("摘要", 3, 1000),
        )

    context = _context(
        screen=screen,
        context_manager=SimpleNamespace(build_for_model_result=build),
        session=SimpleNamespace(
            get_messages=lambda: [], get_compactions=lambda: [], add_compaction=added.append
        ),
    )

    await compact_command_slash.handler(context)

    assert len(added) == 1
    assert "Context compacted" in screen.entries[-1][1]


@pytest.mark.asyncio
async def test_compact_command_reports_failure() -> None:
    """测试压缩失败时显示错误提示。"""

    screen = _Screen()

    async def build(*args, **kwargs) -> ContextBuildResult:
        raise ContextSummaryError("fail")

    context = _context(
        screen=screen,
        context_manager=SimpleNamespace(build_for_model_result=build),
    )

    await compact_command_slash.handler(context)

    assert screen.entries[-1][1] == "Compaction failed"


@pytest.mark.asyncio
async def test_status_command_shows_model_and_context() -> None:
    """测试 /status 展示模型、工作区与上下文信息。"""

    screen = _Screen()
    context = _context(screen=screen, project_dir=Path("/tmp/ws"))

    await status_command_slash.handler(context)

    content = screen.entries[0][1]
    assert "deepseek-v4-pro" in content
    assert "/tmp/ws" in content
    assert "128000" in content


@pytest.mark.asyncio
async def test_copy_command_copies_last_assistant(monkeypatch) -> None:
    """测试 /copy 复制最后一条助手回复。"""

    screen = _Screen()
    copied: list[str] = []
    monkeypatch.setattr(
        "core.commands.copy.copy_text_to_clipboard",
        lambda text: copied.append(text),
    )
    context = _context(
        screen=screen,
        session=SimpleNamespace(
            get_messages=lambda: [
                Message(role="user", content="问题"),
                Message(role="assistant", content="回答内容"),
            ]
        ),
    )

    await copy_command_slash.handler(context)

    assert copied == ["回答内容"]
    assert "Copied" in screen.entries[0][1]


@pytest.mark.asyncio
async def test_copy_command_reports_empty() -> None:
    """测试没有助手回复时提示无可复制内容。"""

    screen = _Screen()
    context = _context(screen=screen)

    await copy_command_slash.handler(context)

    assert "No assistant response" in screen.entries[0][1]


@pytest.mark.asyncio
async def test_clear_command_clears_screen() -> None:
    """测试 /clear 清空对话区。"""

    screen = _Screen()

    await clear_command_slash.handler(_context(screen=screen))

    assert screen.entries == [("tool", "cleared")]


@pytest.mark.asyncio
async def test_quit_command_exits_application() -> None:
    """测试 /quit 退出应用。"""

    screen = _Screen()

    await quit_command_slash.handler(_context(screen=screen))

    assert screen.exited is True


@pytest.mark.asyncio
async def test_export_command_writes_markdown_file(tmp_path: Path) -> None:
    """测试 /export 写出带时间戳的 Markdown 文件。"""

    screen = _Screen()
    context = _context(
        screen=screen,
        project_dir=tmp_path,
        session=SimpleNamespace(
            get_messages=lambda: [
                Message(role="user", content="你好"),
                Message(role="assistant", content="你好呀"),
            ]
        ),
    )

    await export_command_slash.handler(context)

    files = list(tmp_path.glob("conversation-*.md"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "## User" in content and "你好" in content
    assert "## Assistant" in content and "你好呀" in content


@pytest.mark.asyncio
async def test_diff_command_reports_non_git_dir(tmp_path: Path) -> None:
    """测试非 git 目录下 /diff 返回失败提示。"""

    screen = _Screen()

    await diff_command_slash.handler(_context(screen=screen, project_dir=tmp_path))

    assert "git diff failed" in screen.entries[0][1]


@pytest.mark.asyncio
async def test_delete_command_confirms_then_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试 /delete 确认后删除会话并退出。"""

    from core.commands import delete_command_slash

    session_id = "33333333-3333-3333-3333-333333333333"
    screen = _Screen()
    screen.choices = ["yes"]
    context = _context(screen=screen, project_dir=tmp_path)

    # 构造带 session_id 的会话对象与真实 SessionStore 文件
    from core.session_store import SessionStore

    store = SessionStore(tmp_path)
    store.append_message(
        session_id,
        Message(role="user", content="你好"),
    )
    session_path = store._session_path(session_id)
    assert session_path.exists()

    # 替换 _context 里默认的 session
    from dataclasses import replace

    context = replace(
        context,
        session=SimpleNamespace(session_id=session_id, mark_deleted=lambda: None),
    )

    await delete_command_slash.handler(context)

    assert not session_path.exists()
    assert screen.exited is True


@pytest.mark.asyncio
async def test_delete_command_aborts_on_no() -> None:
    """测试 /delete 选择 no 时不删除不退出。"""

    from core.commands import delete_command_slash

    screen = _Screen()
    screen.choices = ["no"]
    context = _context(screen=screen)

    await delete_command_slash.handler(context)

    assert screen.exited is False


@pytest.mark.asyncio
async def test_thinking_toggle_command_switches_state() -> None:
    """测试 /thinking-toggle 切换思考展示状态。"""

    from core.commands import thinking_toggle_command_slash

    screen = _Screen()
    agent_loop = SimpleNamespace(
        thinking_level="high",
        show_thinking=True,
        set_show_thinking=lambda show: setattr(
            agent_loop, "show_thinking", show
        ),
    )
    context = _context(screen=screen, agent_loop=agent_loop)

    await thinking_toggle_command_slash.handler(context)

    assert agent_loop.show_thinking is False
    assert "Thinking display: hidden" in screen.entries[0][1]
