"""解析模型服务端声明的运行能力。"""

from dataclasses import dataclass
from typing import Protocol

from .config import ConfigError, Settings


@dataclass(frozen=True)
class ModelCapabilities:
    """保存上下文管理当前需要的模型能力。"""

    context_window: int
    source: str


class ModelCapabilitiesProvider(Protocol):
    """定义模型客户端可选的能力发现接口。"""

    async def discover_context_window(self) -> int | None:
        """返回服务端声明的上下文窗口，无法发现时返回 None。"""


def extract_context_window(model: object) -> int | None:
    """从 OpenAI-compatible 模型元数据中读取常见窗口字段。"""

    extras = getattr(model, "model_extra", None) or {}
    for name in (
        "context_window",
        "context_length",
        "max_context_length",
        "max_model_len",
        "max_position_embeddings",
    ):
        value = getattr(model, name, None)
        if value is None and isinstance(extras, dict):
            value = extras.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


async def resolve_model_capabilities(
    settings: Settings,
    provider: ModelCapabilitiesProvider,
) -> ModelCapabilities:
    """按显式配置、服务端元数据的顺序解析模型能力。"""

    if settings.context_window is not None:
        return ModelCapabilities(settings.context_window, "configured")

    discovered = await provider.discover_context_window()
    if discovered is not None:
        return ModelCapabilities(discovered, "provider")

    raise ConfigError(
        "The provider does not report this model's context window; "
        "set model.context_window in settings.json"
    )
