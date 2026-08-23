"""Logo 默认实现与起始信息合并的单元测试。"""

from prompt_toolkit.formatted_text import to_plain_text

from core.logo import DefaultLogoProvider


def test_default_logo_renders_ascii_art() -> None:
    """测试默认 Logo 只使用稳定的 ASCII 字符。"""

    logo = DefaultLogoProvider().render()

    text = to_plain_text(logo)

    assert text.isascii()
    assert "EPSILON" in text


def test_default_logo_uses_accent_style() -> None:
    """测试 ASCII Logo 整体使用独立样式类。"""

    logo = DefaultLogoProvider().render()

    assert all(style == "class:logo-accent" for style, _ in logo)
