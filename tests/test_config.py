import json
from pathlib import Path

import pytest

from core.config import ConfigError, McpStdioSettings, Settings, load_settings


def _write_user_settings(tmp_path: Path, data: dict) -> Path:
    """在临时目录写入测试用的用户级 settings.json。"""

    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_project_settings(project_dir: Path, data: dict) -> Path:
    """在项目级 .epsilon 目录写入测试用的项目 settings.json。"""

    path = project_dir / ".epsilon" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _valid_model() -> dict:
    """返回一份合法的最小用户级模型配置。"""

    return {
        "model": {
            "base_url": "https://example.com/v1",
            "model_name": "test-model",
            "api_key": "test-key",
        }
    }


def test_load_settings_reads_required_values(tmp_path: Path) -> None:
    """测试配置加载器能读取三个必填配置项。"""

    user_path = _write_user_settings(tmp_path, _valid_model())

    settings = load_settings(user_config_path=user_path)

    assert settings.base_url == "https://example.com/v1"
    assert settings.model_name == "test-model"
    assert settings.api_key == "test-key"
    assert settings.price is None


def test_load_settings_reads_model_price(tmp_path: Path) -> None:
    """测试可选 model.price 配置被完整读取。"""

    user_path = _write_user_settings(
        tmp_path,
        {
            "model": {
                **_valid_model()["model"],
                "price": {
                    "input": 0.55,
                    "output": 2.19,
                    "cache_read": 0.14,
                },
            }
        },
    )

    settings = load_settings(user_config_path=user_path)

    assert settings.price is not None
    assert settings.price.input == 0.55
    assert settings.price.output == 2.19
    assert settings.price.cache_read == 0.14
    assert settings.price.cache_write is None


def test_load_settings_rejects_price_without_output(tmp_path: Path) -> None:
    """测试 model.price 缺少 output 时配置被拒绝。"""

    user_path = _write_user_settings(
        tmp_path,
        {
            "model": {
                **_valid_model()["model"],
                "price": {"input": 0.55},
            }
        },
    )

    with pytest.raises(ConfigError):
        load_settings(user_config_path=user_path)


def test_load_settings_reads_optional_context_budget(tmp_path: Path) -> None:
    """测试可选上下文预算字段会被完整读取。"""

    user_path = _write_user_settings(
        tmp_path,
        {
            "model": {
                **_valid_model()["model"],
                "context_window": 50000,
                "reserve_tokens": 8000,
                "keep_recent_tokens": 12000,
            }
        },
    )

    settings = load_settings(user_config_path=user_path)

    assert settings.context_window == 50000
    assert settings.reserve_tokens == 8000
    assert settings.keep_recent_tokens == 12000


def test_load_settings_reads_request_timeout(tmp_path: Path) -> None:
    """测试旧请求超时会作为两类超时的兼容默认值。"""

    user_path = _write_user_settings(
        tmp_path,
        {
            "model": {
                **_valid_model()["model"],
                "request_timeout_seconds": 45.5,
            }
        },
    )

    settings = load_settings(user_config_path=user_path)

    assert settings.request_timeout_seconds == 45.5
    assert settings.first_byte_timeout_seconds == 45.5
    assert settings.stream_idle_timeout_seconds == 45.5


def test_load_settings_reads_separate_stream_timeouts(tmp_path: Path) -> None:
    """测试首包和流式空闲超时可以独立配置。"""

    user_path = _write_user_settings(
        tmp_path,
        {
            "model": {
                **_valid_model()["model"],
                "request_timeout_seconds": 45.5,
                "first_byte_timeout_seconds": 12,
                "stream_idle_timeout_seconds": 34,
            }
        },
    )

    settings = load_settings(user_config_path=user_path)

    assert settings.first_byte_timeout_seconds == 12
    assert settings.stream_idle_timeout_seconds == 34


def test_load_settings_reads_optional_stdio_mcp_provider(tmp_path: Path) -> None:
    """测试可选 stdio MCP Provider 配置会被完整读取。"""

    user_path = _write_user_settings(
        tmp_path,
        {
            **_valid_model(),
            "mcp_stdio": {
                "command": "node",
                "arguments": ["server.js", "--readonly"],
                "provider_id": "demo",
            },
        },
    )

    assert load_settings(user_config_path=user_path).mcp_stdio == McpStdioSettings(
        command="node",
        arguments=("server.js", "--readonly"),
        provider_id="demo",
    )


def test_load_settings_rejects_invalid_stdio_mcp_arguments(tmp_path: Path) -> None:
    """测试 MCP 参数必须是 JSON 字符串数组。"""

    user_path = _write_user_settings(
        tmp_path,
        {
            **_valid_model(),
            "mcp_stdio": {
                "command": "node",
                "arguments": "--server",
                "provider_id": "demo",
            },
        },
    )

    with pytest.raises(ConfigError, match="mcp_stdio.arguments"):
        load_settings(user_config_path=user_path)


def test_load_settings_rejects_missing_api_key(tmp_path: Path) -> None:
    """测试缺少 API Key 时会抛出明确的配置异常。"""

    user_path = _write_user_settings(
        tmp_path,
        {
            "model": {
                "base_url": "https://example.com/v1",
                "model_name": "test-model",
            }
        },
    )

    with pytest.raises(ConfigError, match="model.api_key"):
        load_settings(user_config_path=user_path)


def test_load_settings_rejects_invalid_base_url(tmp_path: Path) -> None:
    """测试模型服务地址格式不正确时会抛出配置异常。"""

    user_path = _write_user_settings(
        tmp_path,
        {
            "model": {
                "base_url": "example.com/v1",
                "model_name": "test-model",
                "api_key": "test-key",
            }
        },
    )

    with pytest.raises(ConfigError, match="model.base_url"):
        load_settings(user_config_path=user_path)


def test_load_settings_rejects_missing_user_config(tmp_path: Path) -> None:
    """测试用户配置文件缺失时抛出配置异常。"""

    with pytest.raises(ConfigError, match="User config not found"):
        load_settings(user_config_path=tmp_path / "settings.json")


def test_load_settings_rejects_invalid_json(tmp_path: Path) -> None:
    """测试配置文件不是合法 JSON 时抛出配置异常。"""

    user_path = tmp_path / "settings.json"
    user_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ConfigError, match="not valid JSON"):
        load_settings(user_config_path=user_path)


def test_load_settings_rejects_non_object_json(tmp_path: Path) -> None:
    """测试配置文件不是 JSON 对象时抛出配置异常。"""

    user_path = _write_user_settings(tmp_path, [1, 2, 3])

    with pytest.raises(ConfigError, match="must be a JSON object"):
        load_settings(user_config_path=user_path)


def test_load_settings_defaults_stream_usage_off(tmp_path: Path) -> None:
    """测试默认关闭服务端 usage 采集。"""

    user_path = _write_user_settings(tmp_path, _valid_model())

    assert load_settings(user_config_path=user_path).stream_usage is False


def test_load_settings_reads_stream_usage(tmp_path: Path) -> None:
    """测试 stream_usage 开启时会被正确读取。"""

    user_path = _write_user_settings(
        tmp_path,
        {"model": {**_valid_model()["model"], "stream_usage": True}},
    )

    assert load_settings(user_config_path=user_path).stream_usage is True


def test_load_settings_rejects_invalid_stream_usage(tmp_path: Path) -> None:
    """测试 stream_usage 不是布尔值时抛出配置异常。"""

    user_path = _write_user_settings(
        tmp_path,
        {"model": {**_valid_model()["model"], "stream_usage": "maybe"}},
    )

    with pytest.raises(ConfigError, match="model.stream_usage"):
        load_settings(user_config_path=user_path)


def test_load_settings_defaults_max_tool_rounds(tmp_path: Path) -> None:
    """测试默认不限制交互工具轮次。"""

    user_path = _write_user_settings(tmp_path, _valid_model())

    assert load_settings(user_config_path=user_path).max_tool_rounds is None


def test_load_settings_reads_max_tool_rounds(tmp_path: Path) -> None:
    """测试 max_tool_rounds 会被正确读取。"""

    user_path = _write_user_settings(
        tmp_path,
        {"model": {**_valid_model()["model"], "max_tool_rounds": 5}},
    )

    assert load_settings(user_config_path=user_path).max_tool_rounds == 5


def test_load_settings_rejects_invalid_max_tool_rounds(tmp_path: Path) -> None:
    """测试 max_tool_rounds 不是正整数时抛出配置异常。"""

    user_path = _write_user_settings(
        tmp_path,
        {"model": {**_valid_model()["model"], "max_tool_rounds": 0}},
    )

    with pytest.raises(ConfigError, match="max_tool_rounds"):
        load_settings(user_config_path=user_path)


def test_project_settings_overrides_user_settings(tmp_path: Path) -> None:
    """测试项目级配置按字段覆盖用户级配置。"""

    user_path = _write_user_settings(
        tmp_path,
        {
            "model": {
                **_valid_model()["model"],
                "model_name": "user-model",
                "context_window": 100000,
            }
        },
    )
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_project_settings(
        project_dir,
        {"model": {"model_name": "project-model", "api_key": "project-key"}},
    )

    settings = load_settings(project_dir=project_dir, user_config_path=user_path)

    assert settings.model_name == "project-model"
    assert settings.api_key == "project-key"
    assert settings.base_url == "https://example.com/v1"
    assert settings.context_window == 100000


def test_project_settings_null_field_keeps_user_value(tmp_path: Path) -> None:
    """测试项目级 null 字段表示未设置，不覆盖用户级配置。"""

    user_path = _write_user_settings(
        tmp_path,
        {"model": {**_valid_model()["model"], "model_name": "user-model"}},
    )
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_project_settings(project_dir, {"model": {"model_name": None}})

    settings = load_settings(project_dir=project_dir, user_config_path=user_path)

    assert settings.model_name == "user-model"


def test_project_settings_merges_mcp_stdio(tmp_path: Path) -> None:
    """测试项目级配置可以整体覆盖 mcp_stdio，且不影响用户级 model。"""

    user_path = _write_user_settings(tmp_path, _valid_model())
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_project_settings(
        project_dir,
        {
            "mcp_stdio": {
                "command": "node",
                "arguments": ["project.js"],
                "provider_id": "project-mcp",
            }
        },
    )

    settings = load_settings(project_dir=project_dir, user_config_path=user_path)

    assert settings.mcp_stdio == McpStdioSettings(
        command="node",
        arguments=("project.js",),
        provider_id="project-mcp",
    )
    assert settings.model_name == "test-model"


def test_settings_direct_construction_uses_legacy_timeout_as_fallback() -> None:
    """测试直接构造配置也会统一超时回退。"""

    settings = Settings(
        base_url="https://example.com/v1",
        model_name="test-model",
        api_key="test-key",
        request_timeout_seconds=0.01,
    )

    assert settings.first_byte_timeout_seconds == 0.01
    assert settings.stream_idle_timeout_seconds == 0.01


def test_settings_rejects_invalid_direct_timeout() -> None:
    """测试直接构造配置会校验超时值。"""

    with pytest.raises(ConfigError, match="stream_idle_timeout_seconds"):
        Settings(
            base_url="https://example.com/v1",
            model_name="test-model",
            api_key="test-key",
            stream_idle_timeout_seconds=0,
        )
