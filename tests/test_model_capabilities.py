import pytest

from core.config import ConfigError, Settings
from core.model_capabilities import resolve_model_capabilities


class FakeProvider:
    """返回测试指定的模型上下文窗口。"""

    def __init__(self, context_window: int | None) -> None:
        self.context_window = context_window
        self.calls = 0

    async def discover_context_window(self) -> int | None:
        self.calls += 1
        return self.context_window


@pytest.mark.asyncio
async def test_explicit_context_window_has_priority() -> None:
    """显式配置优先，避免启动时发起不必要的能力查询。"""

    provider = FakeProvider(1_000_000)
    capabilities = await resolve_model_capabilities(
        Settings("https://example.com", "model", "key", context_window=200_000),
        provider,
    )

    assert capabilities.context_window == 200_000
    assert capabilities.source == "configured"
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_context_window_is_discovered_from_provider() -> None:
    """缺少显式配置时使用服务端返回的模型能力。"""

    capabilities = await resolve_model_capabilities(
        Settings("https://example.com", "model", "key"),
        FakeProvider(1_000_000),
    )

    assert capabilities.context_window == 1_000_000
    assert capabilities.source == "provider"


@pytest.mark.asyncio
async def test_unknown_context_window_requires_configuration() -> None:
    """服务端没有元数据时拒绝猜测模型窗口。"""

    with pytest.raises(ConfigError, match="model.context_window"):
        await resolve_model_capabilities(
            Settings("https://example.com", "model", "key"),
            FakeProvider(None),
        )
