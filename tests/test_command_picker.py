"""测试命令补全单选列表组件。"""

import pytest

from prompt_toolkit.completion import Completion

from core.command_picker import CommandPicker


def _completions(names: list[str]) -> list[Completion]:
    """构造补全项列表。"""

    return [
        Completion(name, start_position=-2, display_meta=f"{name} 的描述")
        for name in names
    ]


def test_command_picker_defaults_to_first_item() -> None:
    """测试默认选中第一项。"""

    picker = CommandPicker(_completions(["model", "start-skill"]))

    assert picker.selected.text == "model"


def test_command_picker_move_stays_at_boundaries() -> None:
    """测试光标到达首尾后保持边界。"""

    picker = CommandPicker(_completions(["model", "start-skill", "stop-skill"]))

    picker.move(-1)
    assert picker.selected.text == "model"

    picker.move(10)
    assert picker.selected.text == "stop-skill"


def test_command_picker_renders_name_and_meta() -> None:
    """测试渲染包含命令名与描述。"""

    picker = CommandPicker(_completions(["model"]))

    rendered = "".join(item[1] for item in picker._render())

    assert "/model" in rendered
    assert "model 的描述" in rendered


def test_command_picker_renders_selected_with_prefix() -> None:
    """测试选中项带 › 前缀且使用选中样式。"""

    picker = CommandPicker(_completions(["model", "start-skill"]))

    fragments = picker._render()

    assert fragments[0][0] == "class:approval-selected"
    assert fragments[0][1].startswith("› ")


def test_command_picker_aligns_description_column() -> None:
    """测试描述列左对齐：name 列宽一致，间距拉大。"""

    picker = CommandPicker(
        [
            Completion("model", display_meta="切换模型"),
            Completion("start-skill", display_meta="激活 skill"),
        ]
    )

    fragments = picker._render()

    # 选中行：name 列宽 = max(/model, /start-skill) = 12 + GAP 3
    selected_line = "".join(item[1] for item in fragments if item[1].startswith("›"))
    assert selected_line.startswith("› /model")
    assert "切换模型" in selected_line

    # 未选中行 /start-skill 的描述与选中行描述左对齐（列宽一致）
    all_text = "".join(item[1] for item in fragments)
    lines = [line for line in all_text.split("\n") if line.strip()]
    desc_positions = [
        pos for line in lines for pos in (line.find("激活 skill"), line.find("切换模型"))
        if pos != -1
    ]
    assert len(set(desc_positions)) == 1

    # 未选中行的描述使用独立淡灰样式
    description_styles = [
        item[0]
        for item in fragments
        if "切换模型" in item[1] or "激活 skill" in item[1]
    ]
    assert "class:completion-description" in description_styles


def test_command_picker_has_no_background_style() -> None:
    """测试补全列表无背景（字体落在终端默认背景）。"""

    picker = CommandPicker(_completions(["model"]))

    assert "approval-area" not in picker.window.style


def test_command_picker_scrolls_to_keep_selection_visible() -> None:
    """测试移动到底部时窗口滚动跟随。"""

    picker = CommandPicker(_completions([f"cmd-{index}" for index in range(20)]))

    picker.move(15)

    assert picker.window.vertical_scroll > 0

    picker = CommandPicker(_completions([f"cmd-{index}" for index in range(20)]))
    picker.move(19)
    scroll = picker.window.vertical_scroll

    assert scroll <= 19 < scroll + CommandPicker._VISIBLE_ROWS


def test_command_picker_scroll_uses_item_rows() -> None:
    """测试带描述的候选项仍按固定单行滚动。"""

    picker = CommandPicker(
        [Completion(f"cmd-{index}", display_meta="description") for index in range(10)]
    )

    picker.move(5)

    scroll = picker.window.vertical_scroll
    assert scroll <= 5 < scroll + CommandPicker._VISIBLE_ROWS


def test_command_picker_keeps_scroll_after_window_render() -> None:
    """测试实际写屏后选中项滚动位置不会被 Window 重置。"""

    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition

    picker = CommandPicker(
        [Completion(f"cmd-{index}", display_meta="description") for index in range(20)]
    )
    picker.move(10)
    picker.window.write_to_screen(
        Screen(),
        MouseHandlers(),
        WritePosition(0, 0, 80, CommandPicker._VISIBLE_ROWS),
        "",
        False,
        0,
    )

    scroll = picker.window.vertical_scroll
    assert scroll <= 10 < scroll + CommandPicker._VISIBLE_ROWS
    assert scroll > 0


def test_command_picker_keeps_window_when_moving_back_inside_it() -> None:
    """测试向上回到可见区域时不把选中项跳到列表首行。"""

    picker = CommandPicker(_completions([f"cmd-{index}" for index in range(12)]))

    picker.move(CommandPicker._VISIBLE_ROWS)
    assert picker.window.vertical_scroll == 1

    picker.move(-1)

    assert picker.selected.text == "cmd-7"
    assert picker.window.vertical_scroll == 1


@pytest.mark.asyncio
async def test_command_picker_click_applies_completion() -> None:
    """测试鼠标点击某行应用对应补全。"""

    applied: list[Completion] = []
    picker = CommandPicker(
        _completions(["model", "thinking"]),
        on_apply=applied.append,
    )
    fragments = picker._render()
    handler = None
    for item in fragments:
        if item[1].startswith("  /thinking"):
            handler = item[2]
            break
    assert handler is not None
    handler(None)

    assert len(applied) == 1
    assert applied[0].text == "thinking"


def test_command_picker_update_keeps_selection_and_scroll() -> None:
    """测试增量更新补全列表时保留选中项与滚动位置。"""

    picker = CommandPicker(_completions([f"cmd-{index}" for index in range(20)]))
    picker.move(15)
    scroll_before = picker.window.vertical_scroll

    # 相同的列表内容不重建状态
    picker.update_completions(_completions([f"cmd-{index}" for index in range(20)]))
    assert picker._cursor == 15
    assert picker.window.vertical_scroll == scroll_before

    # 列表收缩时尽量保留选中项，越界则钳制
    picker.update_completions(_completions(["cmd-15", "cmd-16"]))
    assert picker.selected.text == "cmd-15"
    assert picker.window.vertical_scroll == 0

    # 选中项仍在新列表中时精确保持
    picker2 = CommandPicker(_completions(["a", "b", "c", "d"]))
    picker2.move(2)
    picker2.update_completions(_completions(["a", "b", "c", "d", "e"]))
    assert picker2.selected.text == "c"
