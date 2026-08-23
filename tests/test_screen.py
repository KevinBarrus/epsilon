import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import Completion
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.formatted_text import to_plain_text
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.layout.containers import VerticalAlign
from prompt_toolkit.layout.screen import WritePosition
from prompt_toolkit.output import DummyOutput

from core.screen import (
    ChatScreen,
    DraftState,
    SlashCommandCompleter,
)
from core.status import create_status_info
from core.model import ToolCall
from core.tools import ApprovalDecision, ToolDefinition
from core.ui_config import InputLayoutConfig


def _create_screen(tmp_path: Path) -> ChatScreen:
    """创建测试用的全屏界面。"""

    status = create_status_info("test-model", "暂不可查询", tmp_path)
    return ChatScreen(status, inline_mode=False)


def _approval_definition() -> ToolDefinition:
    """构造测试用写工具定义。"""

    return ToolDefinition(
        name="write_file",
        description="写入文件",
        parameters={"type": "object"},
        source="local",
        permission="write",
        idempotent=True,
    )


def _binding_key(binding) -> str:
    """返回绑定的第一个键值，兼容字符串键和枚举键。"""

    key = binding.keys[0]
    return key.value if hasattr(key, "value") else key


def test_chat_screen_uses_full_screen_without_application_mouse_support(tmp_path: Path) -> None:
    """测试兼容全屏界面不再接管终端鼠标事件。"""

    screen = _create_screen(tmp_path)

    assert screen.application.full_screen is True
    assert screen.application.mouse_support() is False


@pytest.mark.asyncio
async def test_approval_replaces_bottom_area_and_restores_layout(tmp_path: Path) -> None:
    """测试审批期间显示在输入区下方并在完成后恢复状态栏。"""

    with create_app_session(output=DummyOutput()):
        screen = _create_screen(tmp_path)
        task = asyncio.create_task(
            screen.request_approval(
                _approval_definition(),
                ToolCall("call-1", "write_file", {"path": "a.txt"}),
            )
        )
        await asyncio.sleep(0)

        assert screen._approval_prompt is not None
        assert screen._bottom_container.children == [screen._approval_prompt.window]
        assert screen._layout.container.children[-1].children[0] is screen._approval_prompt.window

        screen._approval_prompt.confirm()
        result = await task

        assert result.decision == ApprovalDecision.ALLOW_ONCE
        assert screen._approval_prompt is None
        assert screen._bottom_container.children == [screen._status_window]


@pytest.mark.asyncio
async def test_skill_picker_replaces_bottom_area_and_restores_layout(tmp_path: Path) -> None:
    """测试 skill 选择期间显示在输入区下方并在完成后恢复状态栏。"""

    with create_app_session(output=DummyOutput()):
        screen = _create_screen(tmp_path)
        task = asyncio.create_task(
            screen.request_skill_picker([("a", "A 描述", "project")], checked=set())
        )
        await asyncio.sleep(0)

        assert screen._skill_picker is not None
        assert screen._bottom_container.children == [screen._skill_picker.window]

        screen._skill_picker.toggle()   # 勾选 ("a", "project")
        screen._skill_picker.confirm()
        result = await task

        assert result == {("a", "project")}
        assert screen._skill_picker is None
        assert screen._bottom_container.children == [screen._status_window]


def test_chat_screen_renders_status_rows(tmp_path: Path) -> None:
    """测试两行式状态栏：行一左工作区、行二左信息、右模型名。"""

    screen = _create_screen(tmp_path)

    row1_left, row1_right = screen._status_rows()[0]
    row2_left, row2_right = screen._status_rows()[1]

    assert "test-model" in row2_right
    assert "暂不可查询" in row2_left
    assert str(tmp_path) in row1_left
    assert row1_right == ""


def test_chat_screen_renders_runtime_status_message(tmp_path: Path) -> None:
    """测试状态栏行一可以展示运行时降级提示。"""

    screen = _create_screen(tmp_path)
    screen.set_status_message("Session persistence degraded")

    left, _ = screen._status_rows()[0]

    assert "Session persistence degraded" in left


def test_chat_screen_appends_conversation_entries(tmp_path: Path) -> None:
    """测试对话区可以追加并渲染模型内容。"""

    screen = _create_screen(tmp_path)
    index = screen.add_entry("assistant", "测试回复")

    assert screen._render_entry(index) == "测试回复"


@pytest.mark.asyncio
async def test_inline_screen_publishes_only_stable_entries(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """测试行内界面只把稳定条目输出到终端主屏幕。"""

    class EmptyLogo:
        """提供空 Logo，避免干扰历史输出断言。"""

        def render(self) -> str:
            """返回空 Logo。"""

            return ""

    screen = ChatScreen(
        create_status_info("test-model", "n/a", tmp_path),
        logo_provider=EmptyLogo(),
    )
    screen.add_entry("user", "稳定消息")
    active_index = screen.add_active_entry("assistant", "活动回复")

    assert [entry.role for entry in screen.active_entries()] == ["assistant"]
    await screen.flush_history()
    assert "稳定消息" in capsys.readouterr().out

    screen.commit_entry(active_index)
    await screen.flush_history()
    assert "活动回复" in capsys.readouterr().out
    assert len(screen._conversation_content.children) == 0
    assert screen._bottom_container.children == [
        screen._input_tracker,
        screen._status_window,
    ]
    assert screen._conversation_gap in screen.application.layout.container.children


@pytest.mark.asyncio
async def test_running_screen_updates_assistant_stream_immediately(
    tmp_path: Path,
) -> None:
    """测试流式分片立即写入助手控件。"""

    screen = _create_screen(tmp_path)
    screen.application._is_running = True
    index = screen.add_active_entry("assistant", "")

    screen.append_to_entry(index, "```python\nprint")
    screen.append_to_entry(index, "(1)")

    assert ("class:md-code-block", "print(1)") in screen._conversation[index].control.text


def test_assistant_stream_keeps_full_source_and_stable_boundary(tmp_path: Path) -> None:
    """测试流式助手条目独立保存完整原文与稳定前缀边界。"""

    screen = _create_screen(tmp_path)
    index = screen.add_active_entry("assistant", "第一行\n")
    control = screen._conversation[index].control

    screen.append_to_entry(index, "第二行")
    stream = screen._assistant_streams[control]

    assert stream.source == "第一行\n第二行"
    assert stream.stable_source == ""
    assert stream.tail_source == "第一行\n第二行"
    assert screen._render_entry(index) == "第一行\n第二行"


def test_sync_assistant_commit_clears_stream_state(tmp_path: Path) -> None:
    """测试同步提交后不保留已结束的流式状态。"""

    screen = _create_screen(tmp_path)
    index = screen.add_active_entry("assistant", "回复")
    control = screen._conversation[index].control

    assert screen.commit_entry(index) is True
    assert control not in screen._assistant_streams


@pytest.mark.asyncio
async def test_running_screen_finalizes_assistant_highlight_in_background(
    tmp_path: Path,
) -> None:
    """测试助手结束后异步高亮并再写入稳定历史。"""

    screen = _create_screen(tmp_path)
    screen.application._is_running = True
    index = screen.add_active_entry("assistant", "```python\nprint(1)\n```")

    assert screen.commit_entry(index) is True
    assert screen._conversation[index].finalizing is True
    assert screen._conversation[index].committed is False
    await screen.flush_history()

    assert screen._conversation[index].committed is True
    assert ("class:md-tok-builtin", "print") in screen._conversation[index].control.text


def test_inline_clear_keeps_stable_entries_out_of_active_view(tmp_path: Path) -> None:
    """测试行内 /clear 只移除活动条目，不改写稳定历史。"""

    class EmptyLogo:
        """提供空 Logo。"""

        def render(self) -> str:
            """返回空文本。"""

            return ""

    screen = ChatScreen(
        create_status_info("test-model", "n/a", tmp_path),
        logo_provider=EmptyLogo(),
    )
    screen.add_entry("user", "已提交")
    screen.add_active_entry("assistant", "活动内容")

    screen.clear_conversation()

    assert [entry.content for entry in screen.committed_entries()] == ["已提交"]
    assert screen.active_entries() == []


def test_inline_history_user_message_keeps_gray_background(tmp_path: Path) -> None:
    """测试写入终端历史的用户消息仍保留整行灰色背景。"""

    class EmptyLogo:
        """提供空 Logo，避免干扰历史条目。"""

        def render(self) -> str:
            """返回空文本。"""

            return ""

    screen = ChatScreen(
        create_status_info("test-model", "n/a", tmp_path),
        logo_provider=EmptyLogo(),
    )
    index = screen.add_entry("user", "第一行\n第二行")

    fragments = screen._history_fragments(index)
    visible = [(style, text) for style, text in fragments if text != "\n"]

    assert visible
    assert all("class:conversation-user" in style for style, _ in visible)
    assert "第一行" in to_plain_text(fragments)
    assert "第二行" in to_plain_text(fragments)


def test_active_code_block_highlights_after_commit(tmp_path: Path) -> None:
    """测试活动代码块跳过高亮，提交后渲染为最终高亮结果。"""

    screen = _create_screen(tmp_path)
    index = screen.add_active_entry("assistant", "```python\nprint(1)")

    active_fragments = screen._conversation[index].control.text
    assert ("class:md-code-block", "print(1)") in active_fragments
    assert ("class:md-tok-builtin", "print") not in active_fragments

    screen.commit_entry(index)

    committed_fragments = screen._conversation[index].control.text
    assert ("class:md-tok-builtin", "print") in committed_fragments


def test_application_limits_stream_redraw_frequency(tmp_path: Path) -> None:
    """测试流式分片不会无限制触发终端重绘。"""

    screen = _create_screen(tmp_path)

    assert screen.application.min_redraw_interval == pytest.approx(1 / 60)


@pytest.mark.asyncio
async def test_inline_tool_expansion_appends_full_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """测试行内工具结果展开时追加完整内容而不是重写旧输出。"""

    class EmptyLogo:
        """提供空 Logo。"""

        def render(self) -> str:
            """返回空文本。"""

            return ""

    screen = ChatScreen(
        create_status_info("test-model", "n/a", tmp_path),
        logo_provider=EmptyLogo(),
    )
    index = screen.add_entry("tool", "摘要\n第一行\n第二行\n第三行")
    screen.set_tool_result(index, "摘要\n第一行\n第二行\n第三行")
    await screen.flush_history()
    capsys.readouterr()

    screen.toggle_tool_expansion()
    await screen.flush_history()

    assert "第三行" in capsys.readouterr().out


def test_chat_screen_commits_active_entry_only_once(tmp_path: Path) -> None:
    """测试活动条目提交后进入稳定历史，重复提交不会重复生效。"""

    screen = _create_screen(tmp_path)
    stable_index = screen.add_entry("user", "稳定消息")
    active_index = screen.add_active_entry("assistant", "流式回复")

    assert [entry.content for entry in screen.committed_entries()] == [
        "",
        "稳定消息",
    ]
    assert [entry.content for entry in screen.active_entries()] == ["流式回复"]
    assert screen.commit_entry(active_index) is True
    assert screen.commit_entry(active_index) is False
    assert [entry.content for entry in screen.committed_entries()] == [
        "",
        "稳定消息",
        "流式回复",
    ]
    assert screen._conversation[stable_index].committed is True


def test_working_entry_stays_active(tmp_path: Path) -> None:
    """测试临时 working 状态不会进入稳定历史。"""

    screen = _create_screen(tmp_path)
    screen.set_working("thinking")

    assert screen.active_entries()[-1].role == "working"
    assert all(entry.role != "working" for entry in screen.committed_entries())


def test_chat_screen_supports_tool_activity_style(tmp_path: Path) -> None:
    """测试工具活动条目使用独立样式。"""

    screen = _create_screen(tmp_path)
    index = screen.add_entry("tool", "✓ read_file  已读取")

    assert screen._conversation[index].role == "tool"
    assert screen._conversation_content.children[index].style == "class:tool-activity"


def test_tool_entry_style_can_be_updated(tmp_path: Path) -> None:
    """测试工具条目样式可从待执行更新为成功/错误。"""

    screen = _create_screen(tmp_path)
    index = screen.add_entry(
        "tool", "▸ read_file  ...", style="class:tool-pending"
    )

    assert screen._conversation_content.children[index].style == "class:tool-pending"

    screen.set_entry_style(index, "class:tool-success")

    assert screen._conversation_content.children[index].style == "class:tool-success"
    assert screen._conversation[index].style == "class:tool-success"


def test_tool_diff_result_renders_add_and_del_lines(tmp_path: Path) -> None:
    """测试工具 diff 结果中新增行与删除行分别着色。"""

    from core.screen import _render_tool_diff

    fragments = _render_tool_diff(
        "file edited", "-old line\n+new line\n+second"
    )

    assert fragments[0] == ("", "file edited")
    assert ("class:tool-diff-del", "-old line") in fragments
    assert ("class:tool-diff-add", "+new line") in fragments
    assert ("class:tool-diff-add", "+second") in fragments


def test_tool_diff_result_without_diff_keeps_summary(tmp_path: Path) -> None:
    """测试无 diff 内容时只保留摘要行。"""

    from core.screen import _render_tool_diff

    assert _render_tool_diff("file written", "") == [("", "file written")]


def test_streaming_entry_update_does_not_rebuild_conversation_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试连续流式文本只更新目标条目控件。"""

    screen = _create_screen(tmp_path)
    index = screen.add_entry("assistant", "第一段")
    children = screen._conversation_content.children
    control = children[index].content
    sync_calls = 0
    original_sync = screen._sync_conversation

    def track_sync() -> None:
        """记录不应发生的全量布局同步。"""

        nonlocal sync_calls
        sync_calls += 1
        original_sync()

    monkeypatch.setattr(screen, "_sync_conversation", track_sync)

    screen.append_to_entry(index, "第二段")
    screen.append_to_entry(index, "第三段")

    assert sync_calls == 0
    assert screen._conversation_content.children is children
    assert to_plain_text(control.text) == "第一段第二段第三段"


def test_adding_history_entries_syncs_conversation_layout_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试恢复历史时只进行一次对话布局同步。"""

    screen = _create_screen(tmp_path)
    sync_calls = 0
    original_sync = screen._sync_conversation

    def track_sync() -> None:
        """记录批量恢复触发的布局同步次数。"""

        nonlocal sync_calls
        sync_calls += 1
        original_sync()

    monkeypatch.setattr(screen, "_sync_conversation", track_sync)

    screen.add_history_entries(
        [("user", "历史问题"), ("assistant", "历史回答"), ("tool", "工具结果")]
    )

    assert sync_calls == 1
    assert [entry.content for entry in screen._conversation] == [
        "",
        "历史问题",
        "历史回答",
        "工具结果",
    ]


def test_chat_screen_uses_natural_height_for_conversation_and_input(
    tmp_path: Path,
) -> None:
    """测试对话区和输入区不会强制占满额外空间。"""

    screen = _create_screen(tmp_path)

    assert screen.input_area.window.dont_extend_height() is True
    assert screen.input_area.window.height.max == 8


def test_chat_screen_does_not_add_an_implicit_fill_area(tmp_path: Path) -> None:
    """测试根布局不会添加撑大输入框或分散状态栏的隐式填充区。"""

    screen = _create_screen(tmp_path)

    assert screen._layout.container.align is VerticalAlign.JUSTIFY


def test_default_logo_is_first_conversation_entry(tmp_path: Path) -> None:
    """测试新会话 Logo 作为对话区第一条内容显示。"""

    screen = _create_screen(tmp_path)

    assert screen._has_logo() is True
    assert screen._conversation[0].role == "logo"
    assert screen._conversation_container.filter() is True
    assert len(screen._conversation) == 1


def test_logo_merges_startup_info(tmp_path: Path) -> None:
    """测试 Logo 渲染时合并起始信息（操作提示、skill、Context 栏）。"""

    class TestLogo:
        """提供测试 Logo 文本。"""

        def render(self) -> str:
            """返回 Logo 文本。"""

            return "logo"

    screen = ChatScreen(
        create_status_info("test-model", "n/a", tmp_path),
        logo_provider=TestLogo(),
        startup_info_provider=lambda: [
            [("class:startup-hint", "hint-line")],
            [("class:startup-context-header", "[Context]")],
        ],
    )

    rendered = to_plain_text(screen._render_logo())

    lines = rendered.split("\n")
    content_lines = [line for line in lines if line.strip()]

    assert [line.strip() for line in content_lines] == [
        "logo",
        "hint-line",
        "[Context]",
    ]
    assert any(not line.strip() for line in lines[1:-1])

    # 每行按自身宽度居中，而不是使用最长行的统一缩进
    assert len(content_lines[0]) - len(content_lines[0].lstrip()) > (
        len(content_lines[1]) - len(content_lines[1].lstrip())
    )

    # 未提供起始信息时只显示 Logo
    plain_screen = ChatScreen(
        create_status_info("test-model", "n/a", tmp_path),
        logo_provider=TestLogo(),
    )

    assert to_plain_text(plain_screen._render_logo()).lstrip() == "logo"


def test_empty_custom_logo_takes_no_layout_space(tmp_path: Path) -> None:
    """测试自定义空 Logo 不占用布局空间。"""

    class EmptyLogo:
        """渲染空内容的 Logo。"""

        def render(self) -> str:
            """返回空文本。"""

            return ""

    screen = ChatScreen(
        create_status_info("test-model", "n/a", tmp_path),
        logo_provider=EmptyLogo(),
    )

    assert screen._has_logo() is False
    assert screen._conversation == []


def test_input_container_does_not_expand_vertically(tmp_path: Path) -> None:
    """测试输入容器只占自身高度，不填满剩余屏幕。"""

    screen = _create_screen(tmp_path)

    assert screen._input_window.dont_extend_height() is True


@pytest.mark.asyncio
async def test_layout_keeps_empty_input_small_and_moves_status_to_bottom(
    tmp_path: Path,
) -> None:
    """测试空输入保持自然高度，有对话后剩余空间只交给对话视口。"""

    with create_app_session(output=DummyOutput()):
        screen = _create_screen(tmp_path)
        root = screen._layout.container
        empty_sizes = root._divide_heights(WritePosition(0, 0, 100, 40))

        # 根布局：对话区（含 Logo 与输入区）+ 状态栏，中间的 0 是布局间隔。
        assert empty_sizes[1:3] == [0, 2]

        screen.add_entry("user", "用户输入")
        screen.add_entry("assistant", "")
        conversation_sizes = root._divide_heights(WritePosition(0, 0, 100, 40))

        # 对话区包含 Logo 与用户消息后高度增长，状态栏位置不变
        assert conversation_sizes[0] > empty_sizes[0]
        assert conversation_sizes[1:3] == [0, 2]


def test_input_container_has_border_lines_above_and_below(tmp_path: Path) -> None:
    """测试输入区上下各有一条水平线，不使用灰色背景。"""

    screen = _create_screen(tmp_path)

    top, middle, bottom = screen._input_container.children

    assert middle is screen._input_window
    for line in (top, bottom):
        assert line.height == 1
        style = line.style
        assert "input-border" in style


def test_chat_screen_uses_configured_input_max_lines(tmp_path: Path) -> None:
    """测试输入区域使用集中配置的最大行数。"""

    screen = ChatScreen(
        create_status_info("test-model", "暂不可查询", tmp_path),
        input_layout=InputLayoutConfig(
            max_lines=6,
        ),
    )

    assert screen.input_area.window.height.max == 6


def test_input_line_prefix_has_no_padding(tmp_path: Path) -> None:
    """测试输入行前缀为空，文字紧贴最左。"""

    screen = ChatScreen(
        create_status_info("test-model", "暂不可查询", tmp_path),
    )

    assert screen._get_input_line_prefix(0, 0) == []
    assert screen._get_input_line_prefix(1, 1) == []


def test_input_top_border_plain_without_overflow(tmp_path: Path) -> None:
    """测试输入未超行时上边界为纯横线。"""

    screen = ChatScreen(
        create_status_info("test-model", "暂不可查询", tmp_path),
    )

    assert screen._render_input_top_border() == "─" * 4096


def test_input_top_border_shows_more_hint(tmp_path: Path, monkeypatch) -> None:
    """测试输入超行时上边界左侧显示 ↑ n more 提示。"""

    screen = ChatScreen(
        create_status_info("test-model", "暂不可查询", tmp_path),
    )
    monkeypatch.setattr(
        screen.input_area.window,
        "render_info",
        SimpleNamespace(
            ui_content=SimpleNamespace(line_count=10),
            window_height=8,
        ),
    )

    border = screen._render_input_top_border()

    assert border.startswith("─── ↑ 2 more ")
    assert border.endswith("─" * 4096)


def test_input_area_wraps_pasted_long_lines(tmp_path: Path) -> None:
    """测试输入框对长行自动换行显示（粘贴大文本不挤单行）。"""

    screen = _create_screen(tmp_path)

    assert screen.input_area.wrap_lines is True
    screen.input_area.buffer.text = "a" * 200
    assert screen.input_area.buffer.text == "a" * 200


def test_chat_screen_uses_blinking_cursor(tmp_path: Path) -> None:
    """测试输入框启用闪烁光标。"""

    screen = _create_screen(tmp_path)

    assert (
        screen.application.cursor.get_cursor_shape(screen.application)
        == CursorShape.BLINKING_BEAM
    )


def test_chat_screen_restores_terminal_cursor_blink_after_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试绘制结束后恢复终端光标闪烁模式。"""

    screen = _create_screen(tmp_path)
    writes: list[str] = []
    monkeypatch.setattr(screen.application.output, "write_raw", writes.append)
    monkeypatch.setattr(screen.application.output, "flush", lambda: None)

    screen._enable_cursor_blink(screen.application)

    assert writes == ["\x1b[?12h"]


def test_inline_screen_clears_stale_border_once_after_resize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试尺寸变化时只清理活动区上方遗留的一行边框。"""

    screen = ChatScreen(create_status_info("test-model", "n/a", tmp_path))
    size = [80, 24]
    calls: list[object] = []
    monkeypatch.setattr(
        screen.application.output,
        "get_size",
        lambda: SimpleNamespace(columns=size[0], rows=size[1]),
    )
    monkeypatch.setattr(
        screen.application.output,
        "cursor_up",
        lambda amount: calls.append(("up", amount)),
    )
    monkeypatch.setattr(
        screen.application.output,
        "erase_down",
        lambda: calls.append("erase"),
    )
    monkeypatch.setattr(
        screen.application.output,
        "flush",
        lambda: calls.append("flush"),
    )
    monkeypatch.setattr(
        screen.application.renderer,
        "reset",
        lambda: calls.append("reset"),
    )

    screen._clear_stale_viewport_on_resize(screen.application)
    size[0] = 100
    screen._clear_stale_viewport_on_resize(screen.application)

    assert calls == [("up", 1), "erase", "flush", "reset"]


def test_user_entry_uses_full_width_gray_style_without_prefix(
    tmp_path: Path,
) -> None:
    """测试用户消息使用整行灰色背景且不显示角色前缀。"""

    screen = _create_screen(tmp_path)
    screen.add_entry("user", "用户输入")

    assert (
        screen.application.style.get_attrs_for_style_str(
            "class:conversation-user"
        ).bgcolor
        == "343541"
    )
    assert screen._render_entry(1) == "用户输入"


def test_user_entry_renders_n_plus_two_rows_all_gray(
    tmp_path: Path,
) -> None:
    """测试用户消息 n 行内容展示为 n+2 行，单窗口整体灰色背景。"""

    screen = _create_screen(tmp_path)
    index = screen.add_entry("user", "你是谁")

    children = screen._conversation_content.children
    user_window = children[index]

    # 内容 1 行 → 展示 3 行（上下各 1 行留白）
    height = user_window.content.preferred_height(
        100, 10, False, lambda text: len(text)
    )
    assert height == 3
    assert user_window.style == "class:conversation-user"

    # 多行内容同样前后各留 1 行（n+2）
    multi_index = screen.add_entry("user", "第一行\n第二行\n第三行")
    multi_window = screen._conversation_content.children[multi_index]

    multi_height = multi_window.content.preferred_height(
        100, 10, False, lambda text: len(text)
    )
    assert multi_height == 5
    assert multi_window.style == "class:conversation-user"

    # 内容行左右各留一个空格（对齐输出留白）
    rendered = to_plain_text(user_window.content.text)
    lines = [line for line in rendered.split("\n") if line.strip()]
    assert lines == [" 你是谁"]


def test_input_selection_can_be_copied_and_pasted(tmp_path: Path) -> None:
    """测试输入框支持复制选中文本和粘贴剪贴板内容。"""

    screen = _create_screen(tmp_path)
    screen.input_area.text = "复制内容"
    screen.input_area.buffer.cursor_position = 0
    screen.input_area.buffer.start_selection()
    screen.input_area.buffer.cursor_position = len("复制内容")

    screen.copy_input_selection()
    screen.input_area.buffer.cursor_position = len(screen.input_area.text)
    screen.paste_to_input()

    assert screen.application.clipboard.get_data().text == "复制内容"
    assert screen.input_area.text == "复制内容复制内容"


def test_submitted_input_is_saved_to_in_memory_history(tmp_path: Path) -> None:
    """测试有效提交会写入本次运行的输入历史。"""

    screen = ChatScreen(
        create_status_info("test-model", "暂不可查询", tmp_path),
        on_submit=lambda prompt: None,
    )
    screen.input_area.text = "  第一条输入  "
    key_binding = next(
        binding
        for binding in screen._key_bindings.bindings
        if _binding_key(binding) == "c-m" and binding.filter()
    )

    class FakeApplication:
        """避免测试启动真实后台任务。"""

        def create_background_task(self, coroutine):
            coroutine.close()
            return None

    class FakeEvent:
        """提供提交按键处理器所需的最小应用对象。"""

        app = FakeApplication()

    key_binding.handler(FakeEvent())

    assert isinstance(screen.input_area.buffer.history, InMemoryHistory)
    assert list(screen.input_area.buffer.history.load_history_strings()) == [
        "  第一条输入  "
    ]


def test_input_arrows_restore_history_and_move_multiline_cursor(tmp_path: Path) -> None:
    """测试空输入恢复历史，有内容时上下键只移动多行光标。"""

    screen = ChatScreen(
        create_status_info("test-model", "暂不可查询", tmp_path),
        on_submit=lambda prompt: None,
    )

    class FakeApplication:
        """避免测试启动真实后台任务。"""

        def create_background_task(self, coroutine):
            coroutine.close()
            return None

    class SubmitEvent:
        """提供提交按键处理器所需的最小应用对象。"""

        app = FakeApplication()

    submit_binding = next(
        binding
        for binding in screen._key_bindings.bindings
        if _binding_key(binding) == "c-m" and binding.filter()
    )
    screen.input_area.text = "上一条输入"
    submit_binding.handler(SubmitEvent())
    screen.input_area.text = "最新输入"
    submit_binding.handler(SubmitEvent())

    class FakeEvent:
        """提供输入方向键处理器所需的最小事件对象。"""

        current_buffer = screen.input_area.buffer

    def invoke(key: str) -> None:
        binding = next(
            binding
            for binding in screen._key_bindings.bindings
            if _binding_key(binding) == key
            and binding.handler.__name__ == f"move_input_{'up' if key == 'up' else 'down'}"
        )
        binding.handler(FakeEvent())

    invoke("up")
    assert screen.input_area.text == "最新输入"

    screen.input_area.text = "第一行\n第二行"
    screen.input_area.buffer.cursor_position = len(screen.input_area.text)
    invoke("up")
    assert screen.input_area.buffer.document.cursor_position_row == 0
    invoke("down")
    assert screen.input_area.buffer.document.cursor_position_row == 1


def test_chat_screen_accepts_logo_provider(tmp_path: Path) -> None:
    """测试 Logo 接口可以向会话顶部提供内容。"""

    class TestLogo:
        """提供测试 Logo 文本。"""

        def render(self) -> str:
            """返回测试 Logo。"""

            return "epsilon"

    status = create_status_info("test-model", "暂不可查询", tmp_path)
    screen = ChatScreen(status, logo_provider=TestLogo())

    assert to_plain_text(screen._render_logo()).lstrip() == "epsilon"


@pytest.mark.asyncio
async def test_cancel_request_restores_submitted_draft(tmp_path: Path) -> None:
    """测试取消请求后恢复发送前的输入内容和光标位置。"""

    started = asyncio.Event()
    never_finished = asyncio.Event()

    async def handle_submit(prompt: str) -> None:
        """模拟一个持续等待的模型请求。"""

        started.set()
        await never_finished.wait()

    screen = ChatScreen(
        create_status_info("test-model", "暂不可查询", tmp_path),
        on_submit=handle_submit,
    )
    screen._submitted_draft = DraftState("保留这段文字", 2)
    screen.input_area.text = ""
    task = asyncio.create_task(screen._submit("保留这段文字"))
    screen._request_task = task

    await started.wait()
    screen.cancel_request()
    await task

    assert screen.input_area.text == "保留这段文字"
    assert screen.input_area.buffer.cursor_position == 2


class _FakeDocument:
    """提供补全所需的最小文本接口。"""

    def __init__(self, text: str) -> None:
        self.text_before_cursor = text


def test_slash_command_completer_matches_prefix() -> None:
    """测试 / 前缀输入会按命令名前缀匹配补全。"""

    completer = SlashCommandCompleter(
        [
            ("start-skill", "选择并激活 skill"),
            ("stop-skill", "取消已激活的 skill"),
            ("model", "切换模型"),
        ]
    )

    completions = list(completer.get_completions(_FakeDocument("/start"), None))

    assert [completion.text for completion in completions] == ["start-skill"]
    assert to_plain_text(completions[0].display_meta) == "选择并激活 skill"


def test_slash_command_completer_empty_prefix_lists_all() -> None:
    """测试仅输入 / 时列出全部命令。"""

    completer = SlashCommandCompleter([("model", "切换模型"), ("start-skill", "激活")])

    completions = list(completer.get_completions(_FakeDocument("/"), None))

    assert [completion.text for completion in completions] == ["model", "start-skill"]


def test_slash_command_completer_ignores_plain_text() -> None:
    """测试不以 / 开头的输入不触发补全。"""

    completer = SlashCommandCompleter([("model", "切换模型")])

    assert list(completer.get_completions(_FakeDocument("hello"), None)) == []


def test_layout_includes_bottom_area_with_status(tmp_path: Path) -> None:
    """测试根布局包含底部区域，默认显示状态栏。"""

    screen = _create_screen(tmp_path)

    bottom = screen._layout.container.children[-1]

    assert bottom is screen._bottom_container
    assert screen._bottom_container.children == [screen._status_window]


def test_slash_command_completer_prefers_exact_match() -> None:
    """测试输入与命令完全一致时 exact 匹配排在前。"""

    completer = SlashCommandCompleter(
        [("model-switch", "切换配置"), ("model", "切换模型")]
    )

    completions = list(completer.get_completions(_FakeDocument("/model"), None))

    assert [completion.text for completion in completions] == ["model", "model-switch"]


def test_completion_swaps_bottom_area_to_picker(tmp_path: Path) -> None:
    """测试补全出现时底部区域切换为列表，收起后恢复状态栏。"""

    screen = _create_screen(tmp_path)

    assert screen._command_picker is None
    assert screen._bottom_container.children == [screen._status_window]

    screen._on_completions_changed(_FakeBuffer(["model"]))

    assert screen._command_picker is not None
    assert screen._bottom_container.children == [screen._command_picker.window]

    screen._on_completions_changed(_FakeBuffer([]))

    assert screen._command_picker is None
    assert screen._bottom_container.children == [screen._status_window]


class _FakeCompleteState:
    """模拟补全状态，completions 为文本列表。"""

    def __init__(self, texts: list[str]) -> None:
        self.completions = [
            Completion(text, start_position=0) for text in texts
        ]


class _FakeBuffer:
    """模拟带补全状态的输入缓冲区。"""

    def __init__(self, texts: list[str]) -> None:
        self.complete_state = _FakeCompleteState(texts)


def test_selected_completion_uses_command_picker(tmp_path: Path) -> None:
    """测试选中补全项来自补全列表组件。"""

    screen = _create_screen(tmp_path)
    buffer = screen.input_area.buffer

    assert screen._selected_completion(buffer) is None

    screen._on_completions_changed(_FakeBuffer(["model", "model-switch"]))

    assert screen._selected_completion(buffer).text == "model"

    screen._command_picker.move(1)

    assert screen._selected_completion(buffer).text == "model-switch"


def test_input_text_changed_restarts_completion_after_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试删除字符后重新触发命令补全。"""

    screen = _create_screen(tmp_path)
    started: list[bool] = []
    monkeypatch.setattr(
        screen.input_area.buffer,
        "start_completion",
        lambda: started.append(True),
    )

    screen._last_input_length = 4   # 之前是 /mod
    screen.input_area.buffer.complete_state = None
    screen.input_area.buffer.text = "/mo"
    screen._on_input_text_changed(screen.input_area.buffer)

    assert started == [True]
    assert screen._last_input_length == 3


def test_input_text_changed_skips_when_text_grows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试插入字符（文本变长）不重复触发补全，交给 complete_while_typing。"""

    screen = _create_screen(tmp_path)
    started: list[bool] = []
    monkeypatch.setattr(
        screen.input_area.buffer,
        "start_completion",
        lambda: started.append(True),
    )

    screen._last_input_length = 3
    screen.input_area.buffer.text = "/mod"
    screen._on_input_text_changed(screen.input_area.buffer)

    assert started == []


def test_input_text_changed_skips_without_slash_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试不以 / 开头的输入不触发补全。"""

    screen = _create_screen(tmp_path)
    started: list[bool] = []
    monkeypatch.setattr(
        screen.input_area.buffer,
        "start_completion",
        lambda: started.append(True),
    )

    screen._last_input_length = 5
    screen.input_area.buffer.text = "hello"
    screen._on_input_text_changed(screen.input_area.buffer)

    assert started == []


def test_input_text_changed_dismisses_picker_when_slash_removed(
    tmp_path: Path,
) -> None:
    """测试删除到不以 / 开头时收起残留的补全列表并恢复状态栏。"""

    screen = _create_screen(tmp_path)

    # 先有补全列表显示
    screen._on_completions_changed(_FakeBuffer(["model"]))
    assert screen._command_picker is not None
    assert screen._bottom_container.children == [screen._command_picker.window]

    # 删除到空文本（不再以 / 开头）
    screen._last_input_length = 1
    screen.input_area.buffer.text = ""
    screen._on_input_text_changed(screen.input_area.buffer)

    assert screen._command_picker is None
    assert screen._bottom_container.children == [screen._status_window]


def test_input_text_changed_keeps_picker_while_slash_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试删除后仍以 / 开头时（例如 /mod -> /m）列表保留由补全状态维护。"""

    screen = _create_screen(tmp_path)
    started: list[bool] = []
    monkeypatch.setattr(
        screen.input_area.buffer,
        "start_completion",
        lambda: started.append(True),
    )

    screen._on_completions_changed(_FakeBuffer(["model"]))
    assert screen._command_picker is not None

    screen._last_input_length = 4
    screen.input_area.buffer.text = "/m"
    screen._on_input_text_changed(screen.input_area.buffer)

    # 仍以 / 开头：不主动收起，交由补全状态维护
    assert screen._command_picker is not None
    assert started == [True]


def test_status_model_name_uses_provider(tmp_path: Path) -> None:
    """测试状态栏模型名优先使用动态 provider 的值。"""

    provider_value = "dynamic-model"
    screen = ChatScreen(
        create_status_info("static-model", "n/a", tmp_path),
        model_name_provider=lambda: provider_value,
    )

    fragments = screen._status_rows()
    row2_right = fragments[1][1]

    assert "dynamic-model" in row2_right
    assert "static-model" not in row2_right


def test_status_model_name_falls_back_to_status(tmp_path: Path) -> None:
    """测试未提供 provider 时使用状态对象里的模型名。"""

    screen = ChatScreen(create_status_info("static-model", "n/a", tmp_path))

    row2_right = screen._status_rows()[1][1]

    assert "static-model" in row2_right


def test_status_provider_and_thinking_level_appear(tmp_path: Path) -> None:
    """测试状态栏模型行显示厂商名与推理强度。"""

    screen = ChatScreen(
        create_status_info("deepseek-v4-pro", "n/a", tmp_path),
        provider_name_provider=lambda: "deepseek",
        thinking_level_provider=lambda: "high",
    )

    row2_right = screen._status_rows()[1][1]

    assert "(deepseek) deepseek-v4-pro · high" in row2_right


def test_status_balance_uses_provider(tmp_path: Path) -> None:
    """测试状态栏信息行缺省时优先使用动态余额。"""

    provider_value = "9.99 CNY"
    screen = ChatScreen(
        create_status_info("test-model", "unavailable", tmp_path),
        balance_text_provider=lambda: provider_value,
    )

    row2_left = screen._status_rows()[1][0]

    assert "9.99 CNY" in row2_left
    assert "unavailable" not in row2_left


def test_status_balance_falls_back_to_status(tmp_path: Path) -> None:
    """测试未提供 provider 时使用状态对象里的余额。"""

    screen = ChatScreen(create_status_info("test-model", "2.00 CNY", tmp_path))

    row2_left = screen._status_rows()[1][0]

    assert "2.00 CNY" in row2_left


def test_status_copy_hint_provider(tmp_path: Path) -> None:
    """测试状态栏行一右侧显示复制提示，默认空。"""

    screen = ChatScreen(create_status_info("test-model", "n/a", tmp_path))
    assert screen._status_rows()[0][1] == ""

    hint_screen = ChatScreen(
        create_status_info("test-model", "n/a", tmp_path),
        copy_hint_provider=lambda: "Copied 42 chars to clipboard",
    )
    row1_right = hint_screen._status_rows()[0][1]

    assert row1_right == "Copied 42 chars to clipboard"


def test_user_message_renders_markdown(tmp_path: Path) -> None:
    """测试用户消息中的加粗等 Markdown 标记被渲染。"""

    screen = _create_screen(tmp_path)
    index = screen.add_entry("user", "**加粗** 与 `code`")

    fragments = screen._conversation[index].control.text
    assert ("class:md-bold", " 加粗") in fragments
    assert ("class:md-code", "code") in fragments


def test_user_message_keeps_padding_with_markdown(tmp_path: Path) -> None:
    """测试用户消息渲染 Markdown 后仍保留左右留白与上下空行。"""

    screen = _create_screen(tmp_path)
    index = screen.add_entry("user", "你好")

    children = screen._conversation_content.children
    user_window = children[index]

    height = user_window.content.preferred_height(
        100, 10, False, lambda text: len(text)
    )
    assert height == 3
    assert user_window.style == "class:conversation-user"

    rendered = to_plain_text(user_window.content.text)
    lines = [line for line in rendered.split("\n") if line.strip()]
    assert lines == [" 你好"]


def test_completions_changed_does_not_rebuild_picker(tmp_path: Path) -> None:
    """测试补全回调重复触发时复用同一个选择器，保留选中状态。"""

    from types import SimpleNamespace

    from prompt_toolkit.completion import Completion

    def fake_buffer(completions):
        return SimpleNamespace(
            complete_state=SimpleNamespace(completions=completions)
        )

    screen = _create_screen(tmp_path)

    screen._on_completions_changed(
        fake_buffer([Completion("model"), Completion("mcp")])
    )
    first_picker = screen._command_picker
    assert first_picker is not None

    # 模拟 prompt_toolkit 每 0.3 秒重复触发：内容相同的回调不重建
    screen._on_completions_changed(
        fake_buffer([Completion("model"), Completion("mcp")])
    )
    assert screen._command_picker is first_picker

    # 移动选中后重复回调仍保留位置
    first_picker.move(1)
    screen._on_completions_changed(
        fake_buffer([Completion("model"), Completion("mcp")])
    )
    assert screen._command_picker is first_picker
    assert first_picker.selected.text == "mcp"


def test_working_indicator_renders_spinner_and_elapsed(tmp_path: Path) -> None:
    """测试 working 提示显示在对话区并包含 spinner 与耗时。"""

    screen = _create_screen(tmp_path)
    screen.set_working("thinking")

    text = screen._working_text()

    from core.screen import _WORKING_FRAMES
    assert text[0] in _WORKING_FRAMES
    assert "thinking" in text
    assert "s" in text
    assert any(entry.role == "working" for entry in screen._conversation)
    assert "thinking" not in screen._status_rows()[1][0]

    screen.set_working(None)

    assert screen._working_text() == ""
    assert not any(entry.role == "working" for entry in screen._conversation)


def test_tool_result_folds_long_output(tmp_path: Path) -> None:
    """测试超长工具输出折叠并显示省略提示。"""

    from prompt_toolkit.formatted_text import to_plain_text

    screen = _create_screen(tmp_path)
    long_output = "\n".join(f"line {i}" for i in range(20))
    index = screen.add_entry("tool", "")

    screen.set_tool_result(index, long_output)

    text = to_plain_text(screen._conversation[index].control.text)
    assert "line 0" in text
    assert "line 19" not in text
    assert "12 more lines (ctrl+o to expand)" in text


def test_tool_result_expands_on_toggle(tmp_path: Path) -> None:
    """测试 ctrl+o 切换后工具输出完整展开。"""

    from prompt_toolkit.formatted_text import to_plain_text

    screen = _create_screen(tmp_path)
    long_output = "\n".join(f"line {i}" for i in range(20))
    index = screen.add_entry("tool", "")

    screen.set_tool_result(index, long_output)
    screen.toggle_tool_expansion()

    text = to_plain_text(screen._conversation[index].control.text)
    assert "line 19" in text
    assert "more lines" not in text

    # 再切换回折叠
    screen.toggle_tool_expansion()
    text = to_plain_text(screen._conversation[index].control.text)
    assert "more lines" in text


def test_tool_result_keeps_status_and_name_when_folded(tmp_path: Path) -> None:
    """测试成功或失败结果折叠时仍保留状态和工具名称。"""

    from prompt_toolkit.formatted_text import to_plain_text

    screen = _create_screen(tmp_path)
    long_output = "\n".join(f"line {i}" for i in range(20))
    index = screen.add_entry("tool", "")

    screen.set_tool_result(index, "✗ run_command\n" + long_output)

    text = to_plain_text(screen._conversation[index].control.text)
    assert "✗ run_command" in text
    assert "line 19" not in text
    assert "more lines" in text

    screen.toggle_tool_expansion()
    text = to_plain_text(screen._conversation[index].control.text)
    assert "line 19" in text


def test_tool_result_short_output_no_folding(tmp_path: Path) -> None:
    """测试短工具输出不折叠，diff 红绿仍生效。"""

    screen = _create_screen(tmp_path)
    index = screen.add_entry("tool", "")

    screen.set_tool_result(index, "file edited\n-old\n+new")

    fragments = screen._conversation[index].control.text
    assert ("class:tool-diff-del", "-old") in fragments
    assert ("class:tool-diff-add", "+new") in fragments
