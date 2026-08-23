"""对话区 Markdown 基础渲染的单元测试。"""

from prompt_toolkit.formatted_text import to_plain_text

from core.markdown import render_inline, render_markdown


def _plain(text: str) -> str:
    """把渲染结果还原为纯文本。"""

    return to_plain_text(render_markdown(text))


def test_heading_uses_heading_style() -> None:
    """测试标题行使用标题样式并去掉井号前缀。"""

    fragments = render_markdown("# 标题")

    assert fragments == [("class:md-heading", "标题")]


def test_bold_and_italic_inline_styles() -> None:
    """测试粗体和斜体使用独立样式。"""

    fragments = render_markdown("这是**粗体**和*斜体*文本")

    assert ("class:md-bold", "粗体") in fragments
    assert ("class:md-italic", "斜体") in fragments
    assert _plain("这是**粗体**和*斜体*文本") == "这是粗体和斜体文本"


def test_unordered_list_keeps_marker() -> None:
    """测试无序列表带圆点前缀。"""

    fragments = render_markdown("- 列表项")

    assert ("", "• ") in fragments
    assert _plain("- 列表项") == "• 列表项"


def test_fenced_code_block_uses_code_style() -> None:
    """测试围栏代码块显示语言名并按 token 高亮。"""

    fragments = render_markdown("```python\nprint(1)\n```")

    assert ("class:md-code-lang", "python\n") in fragments
    assert ("class:md-tok-builtin", "print") in fragments
    assert ("class:md-tok-number", "1") in fragments
    assert not any(text == "```python" for _, text in fragments)


def test_unclosed_code_block_stays_in_code_style() -> None:
    """测试流式输出未闭合的代码块保持代码样式（容错）。"""

    fragments = render_markdown("```python\nprint(1)")

    assert ("class:md-tok-builtin", "print") in fragments


def test_streaming_unclosed_code_block_skips_syntax_highlighting() -> None:
    """测试流式未闭合代码块只使用基础样式，提交后再高亮。"""

    fragments = render_markdown("```python\nprint(1)", streaming=True)

    assert ("class:md-code-block", "print(1)") in fragments
    assert ("class:md-tok-builtin", "print") not in fragments


def test_code_block_does_not_duplicate_line_breaks() -> None:
    """测试代码围栏内的换行只由代码块渲染器输出一次。"""

    text = "说明\n```python\ndef hello():\n    return 1\n```\n结束"

    assert _plain(text) == "说明\npython\ndef hello():\n    return 1\n结束"


def test_streaming_unclosed_code_block_does_not_accumulate_blank_lines() -> None:
    """测试未闭合代码围栏在流式过程中不制造额外空行。"""

    text = "```python\ndef hello():\n    return 1"

    assert _plain(text) == "python\ndef hello():\n    return 1"


def test_code_block_highlights_javascript() -> None:
    """测试 JavaScript 代码块同样按 token 高亮。"""

    fragments = render_markdown("```javascript\nconst x = 1;\n```")

    assert ("class:md-code-lang", "javascript\n") in fragments
    assert ("class:md-tok-keyword", "const") in fragments


def test_code_block_unknown_language_uses_base_style() -> None:
    """测试不认识的代码语言使用基础代码样式。"""

    fragments = render_markdown("```nosuchlang\nhello world\n```")

    assert ("class:md-code-lang", "nosuchlang\n") in fragments
    assert ("class:md-code-block", "hello world") in fragments


def test_quote_prefix_and_style() -> None:
    """测试引用行带竖线前缀和独立样式。"""

    fragments = render_markdown("> 引用内容")

    assert fragments == [("class:md-quote", "▍ "), ("", "引用内容")]


def test_quote_renders_inline_markup() -> None:
    """测试引用行内粗体与行内代码被解析。"""

    fragments = render_markdown("> **重点** 与 `code`")

    assert ("class:md-bold", "重点") in fragments
    assert ("class:md-code", "code") in fragments


def test_horizontal_rule_replaces_dashes() -> None:
    """测试分隔线渲染为一条水平线。"""

    fragments = render_markdown("---")

    assert fragments == [("class:md-hr", "─" * 20)]


def test_plain_text_unchanged() -> None:
    """测试普通文本渲染后内容不变。"""

    assert _plain("你好，这是普通文本。") == "你好，这是普通文本。"


def test_ordered_list_keeps_marker() -> None:
    """测试有序列表保留序号语义。"""

    assert _plain("1. 第一步") == "· 第一步"


def test_unclosed_bold_shows_as_plain_text() -> None:
    """测试未闭合的粗体标记被忽略，只保留文字。"""

    assert _plain("未闭合**粗体") == "未闭合粗体"


def test_unclosed_backtick_stripped() -> None:
    """测试未闭合的行内代码标记被忽略，只保留文字。"""

    assert _plain("使用 `git status 查看") == "使用 git status 查看"


def test_lone_asterisk_kept() -> None:
    """测试普通文本中的单个星号（如乘法）不被移除。"""

    assert _plain("a * b = c") == "a * b = c"


def test_inline_renders_bold_and_italic() -> None:
    """测试行内渲染函数分别处理粗体和斜体。"""

    fragments = render_inline("a **b** c")

    assert ("class:md-bold", "b") in fragments


def test_multiline_renders_with_line_breaks() -> None:
    """测试多行内容行与行之间插入换行。"""

    fragments = render_markdown("第一行\n第二行\n第三行")

    assert _plain("第一行\n第二行\n第三行") == "第一行\n第二行\n第三行"


def test_ordered_list_lines_break() -> None:
    """测试有序列表每项独立成行。"""

    plain = _plain("1. 第一点\n2. 第二点")

    assert plain == "· 第一点\n· 第二点"


def test_inline_code_uses_code_style() -> None:
    """测试行内代码使用代码样式并去掉反引号。"""

    fragments = render_markdown("运行 `uv run pytest` 即可")

    assert ("class:md-code", "uv run pytest") in fragments
    assert _plain("运行 `uv run pytest` 即可") == "运行 uv run pytest 即可"


def test_link_uses_link_style_and_shows_text() -> None:
    """测试链接只显示文字并使用链接样式。"""

    fragments = render_markdown("访问 [官网](https://example.com) 查看")

    assert ("class:md-link", "官网") in fragments
    assert _plain("访问 [官网](https://example.com) 查看") == "访问 官网 查看"


def test_unclosed_link_stays_plain() -> None:
    """测试流式输出未闭合的链接保持原样。"""

    assert _plain("[官网](https://ex") == "[官网](https://ex"


def test_table_renders_aligned_with_borders() -> None:
    """测试表格渲染为 │ 边框对齐，表头加粗，分隔行跳过。"""

    markdown = "| 名称 | 数量 |\n| --- | --- |\n| 苹果 | 3 |\n| 香蕉 | 10 |"

    fragments = render_markdown(markdown)
    text = "\n".join(content for _, content in fragments)
    styles = [style for style, _ in fragments]

    assert styles[0] == "class:md-table-header"
    assert "│ 名称 │ 数量 │" in text
    assert "│ 苹果 │ 3    │" in text
    assert "│ 香蕉 │ 10   │" in text
    assert "---" not in text


def test_table_flows_with_streaming_rows() -> None:
    """测试流式输出时表格逐行累积渲染（容错）。"""

    fragments = render_markdown("| a | b |\n| --- | --- |\n| 1 | 2 |")

    assert ("class:md-table-header", "│ a │ b │") in fragments

    # 表格未结束时（最后一行缺失）也能渲染已收集的行
    partial = render_markdown("| a | b |\n| --- | --- |")

    assert any("a" in content for _, content in partial)


def test_table_interrupted_by_plain_text() -> None:
    """测试表格后跟普通文本时正常收尾。"""

    fragments = render_markdown("| a |\n| 1 |\n结束")
    text = "\n".join(content for _, content in fragments)

    assert "│ a │" in text
    assert "结束" in text
