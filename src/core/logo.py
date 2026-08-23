"""定义会话 Logo 的可替换渲染接口与默认实现。"""

from typing import Protocol

from prompt_toolkit.formatted_text import AnyFormattedText

# 使用普通 ASCII 字符，避免不同终端对 Unicode 方块宽度和字形支持不一致
_LOGO_ART = (
    r"  _____ ____  ____ ___ _     ___  _   _",
    r" | ____|  _ \|  _ \_ _| |   / _ \| \ | |",
    r" |  _| | |_) | |_) | || |  | | | |  \| |",
    r" | |___|  __/|  __/| || |__| |_| | |\  |",
    r" |_____|_|   |_|  |___|_____\___/|_| \_|",
    "                 EPSILON",
)




class LogoProvider(Protocol):
    """Logo 渲染实现需要遵循的接口。"""

    def render(self) -> AnyFormattedText:
        """返回要显示在会话顶部的格式化文本。"""


class DefaultLogoProvider:
    """默认 Logo：ASCII EPSILON 字样，使用独立样式类以便主题调整。"""

    def render(self) -> AnyFormattedText:
        """渲染 ASCII Logo，后续可替换为专属 Logo。"""

        return [("class:logo-accent", "\n".join(_LOGO_ART))]
