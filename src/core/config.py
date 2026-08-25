"""加载并校验 epsilon 的运行配置。"""

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


class ConfigError(ValueError):
    """配置文件缺失或配置项不合法时抛出的异常。"""


@dataclass(frozen=True)
class McpStdioSettings:
    """描述一个可选 stdio MCP Provider 的启动配置。"""

    command: str
    arguments: tuple[str, ...]
    provider_id: str


@dataclass(frozen=True)
class ModelPrice:
    """模型每百万 token 的美元单价，cache_read 缺省时视为无缓存计费。"""

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None


@dataclass(frozen=True)
class Settings:
    """模型客户端运行所需的最小配置。"""

    base_url: str
    model_name: str
    api_key: str
    context_window: int = 100_000
    reserve_tokens: int = 16_000
    keep_recent_tokens: int = 20_000
    request_timeout_seconds: float = 120.0
    first_byte_timeout_seconds: float | None = None
    stream_idle_timeout_seconds: float | None = None
    mcp_stdio: McpStdioSettings | None = None
    stream_usage: bool = False
    max_tool_rounds: int | None = None
    price: ModelPrice | None = None

    def __post_init__(self) -> None:
        """统一超时默认值并校验直接构造的配置。"""

        first_byte_timeout = (
            self.request_timeout_seconds
            if self.first_byte_timeout_seconds is None
            else self.first_byte_timeout_seconds
        )
        stream_idle_timeout = (
            self.request_timeout_seconds
            if self.stream_idle_timeout_seconds is None
            else self.stream_idle_timeout_seconds
        )
        for name, value in (
            ("request_timeout_seconds", self.request_timeout_seconds),
            ("first_byte_timeout_seconds", first_byte_timeout),
            ("stream_idle_timeout_seconds", stream_idle_timeout),
        ):
            if value <= 0:
                raise ConfigError(f"{name} must be > 0")
        if self.max_tool_rounds is not None and self.max_tool_rounds <= 0:
            raise ConfigError("max_tool_rounds must be > 0")
        object.__setattr__(self, "first_byte_timeout_seconds", first_byte_timeout)
        object.__setattr__(self, "stream_idle_timeout_seconds", stream_idle_timeout)


def load_settings(
    project_dir: Path | None = None,
    user_config_path: Path | None = None,
) -> Settings:
    """从用户级与项目级 settings.json 读取配置并完成基础校验。

    用户级配置（~/.epsilon/settings.json）提供默认值；项目级配置
    （<项目目录>/.epsilon/settings.json）按字段覆盖用户级配置。
    """

    user_path = (user_config_path or default_user_config_path()).resolve()
    if not user_path.is_file():
        raise ConfigError(
            f"User config not found: {user_path}\n"
            "Run epsilon to set up, or create the file manually"
        )
    merged_data = _read_config_json(user_path)
    project_path = _project_config_path(project_dir)
    if project_path.is_file():
        merged_data = _merge_configs(merged_data, _read_config_json(project_path))
    return _settings_from_data(merged_data)


def default_user_config_path() -> Path:
    """返回用户级配置的默认位置。"""

    return Path.home() / ".epsilon" / "settings.json"


def _project_config_path(project_dir: Path | None) -> Path:
    """返回项目级配置的位置，未指定项目目录时使用当前工作目录。"""

    root = (project_dir or Path.cwd()).resolve()
    return root / ".epsilon" / "settings.json"


def _read_config_json(path: Path) -> dict:
    """读取并解析 settings.json，非法 JSON 或非对象时报配置错误。"""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Failed to read config file: {path}") from exc
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must be a JSON object: {path}")
    return data


def _merge_configs(user_data: dict, project_data: dict) -> dict:
    """项目级配置按字段覆盖用户级配置，null 字段表示未设置不参与覆盖。"""

    merged = dict(user_data)
    project_model = project_data.get("model")
    if isinstance(project_model, dict):
        user_model = merged.get("model")
        if not isinstance(user_model, dict):
            user_model = {}
        merged["model"] = {
            **user_model,
            **{name: value for name, value in project_model.items() if value is not None},
        }
    if "mcp_stdio" in project_data and project_data["mcp_stdio"] is not None:
        merged["mcp_stdio"] = project_data["mcp_stdio"]
    return merged


def _settings_from_data(data: dict) -> Settings:
    """从合并后的配置字典构造 Settings，并完成必填项与取值校验。"""

    model = data.get("model")
    if not isinstance(model, dict):
        raise ConfigError("model must be an object")
    base_url = _required_value(model.get("base_url"), "model.base_url")
    model_name = _required_value(model.get("model_name"), "model.model_name")
    api_key = _required_value(model.get("api_key"), "model.api_key")
    context_window = _optional_int(
        model.get("context_window"),
        100_000,
        "model.context_window",
    )
    reserve_tokens = _optional_int(
        model.get("reserve_tokens"),
        16_000,
        "model.reserve_tokens",
    )
    keep_recent_tokens = _optional_int(
        model.get("keep_recent_tokens"),
        20_000,
        "model.keep_recent_tokens",
    )
    request_timeout_seconds = _optional_float(
        model.get("request_timeout_seconds"),
        120.0,
        "model.request_timeout_seconds",
    )
    first_byte_timeout_seconds = _optional_float(
        model.get("first_byte_timeout_seconds"),
        None,
        "model.first_byte_timeout_seconds",
    )
    stream_idle_timeout_seconds = _optional_float(
        model.get("stream_idle_timeout_seconds"),
        None,
        "model.stream_idle_timeout_seconds",
    )
    stream_usage = _optional_bool(
        model.get("stream_usage"),
        False,
        "model.stream_usage",
    )
    max_tool_rounds = _optional_int(
        model.get("max_tool_rounds"),
        None,
        "model.max_tool_rounds",
    )
    price = _optional_model_price(model.get("price"))
    if context_window <= 0:
        raise ConfigError("model.context_window must be > 0")
    if reserve_tokens < 0 or reserve_tokens >= context_window:
        raise ConfigError("model.reserve_tokens must be < model.context_window")
    if keep_recent_tokens <= 0:
        raise ConfigError("model.keep_recent_tokens must be > 0")
    _validate_base_url(base_url)

    return Settings(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        context_window=context_window,
        reserve_tokens=reserve_tokens,
        keep_recent_tokens=keep_recent_tokens,
        request_timeout_seconds=request_timeout_seconds,
        first_byte_timeout_seconds=first_byte_timeout_seconds,
        stream_idle_timeout_seconds=stream_idle_timeout_seconds,
        mcp_stdio=_optional_mcp_stdio_settings(data),
        stream_usage=stream_usage,
        max_tool_rounds=max_tool_rounds,
        price=price,
    )


def _optional_model_price(value: object) -> ModelPrice | None:
    """解析可选的 model.price 配置，缺省返回 None。"""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError("model.price must be an object")
    input_price = value.get("input")
    output_price = value.get("output")
    if not isinstance(input_price, (int, float)) or not isinstance(
        output_price, (int, float)
    ):
        raise ConfigError("model.price.input and model.price.output are required")
    cache_read = value.get("cache_read")
    cache_write = value.get("cache_write")
    return ModelPrice(
        input=float(input_price),
        output=float(output_price),
        cache_read=float(cache_read) if isinstance(cache_read, (int, float)) else None,
        cache_write=float(cache_write) if isinstance(cache_write, (int, float)) else None,
    )


def _required_value(value: object, name: str) -> str:
    """读取必填配置项，避免把空配置传给模型客户端。"""

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing required setting: {name}")
    return value.strip()


def _optional_int(value: object, default: int, name: str) -> int:
    """读取可选整数配置，未设置时使用默认值。"""

    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _optional_float(
    value: object,
    default: float | None,
    name: str,
) -> float | None:
    """读取可选小数配置，未设置时使用默认值。"""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _optional_bool(value: object, default: bool, name: str) -> bool:
    """读取可选布尔配置，未设置时使用默认值。"""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ConfigError(f"{name} must be a boolean")


def _optional_mcp_stdio_settings(data: dict) -> McpStdioSettings | None:
    """读取可选 stdio MCP 配置，并校验参数数组。"""

    mcp = data.get("mcp_stdio")
    if mcp is None:
        return None
    if not isinstance(mcp, dict):
        raise ConfigError("mcp_stdio must be an object")
    command = mcp.get("command")
    arguments = mcp.get("arguments")
    provider_id = mcp.get("provider_id")
    if all(value is None for value in (command, arguments, provider_id)):
        return None
    if not isinstance(arguments, list) or not all(
        isinstance(argument, str) for argument in arguments
    ):
        raise ConfigError("mcp_stdio.arguments must be an array of strings")
    return McpStdioSettings(
        command=_required_value(command, "mcp_stdio.command"),
        arguments=tuple(arguments),
        provider_id=_required_value(provider_id, "mcp_stdio.provider_id"),
    )


def _validate_base_url(base_url: str) -> None:
    """确保模型服务地址使用可识别的 HTTP 协议。"""

    parsed_url = urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ConfigError("model.base_url must be a valid http or https URL")
