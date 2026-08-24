"""提供工具参数读取辅助函数。"""

from ..model import ToolCall


def string_argument(tool_call: ToolCall, name: str) -> str:
    """读取必填的非空字符串参数。"""

    value = tool_call.arguments.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def text_argument(tool_call: ToolCall, name: str) -> str:
    """读取允许为空的文本参数。"""

    value = tool_call.arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def optional_path(tool_call: ToolCall) -> str:
    """读取可选路径参数，未提供时使用工作区根目录。"""

    value = tool_call.arguments.get("path", ".")
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a non-empty string")
    return value


def optional_positive_integer(
    tool_call: ToolCall,
    name: str,
    default: int,
) -> int:
    """读取可选正整数参数，未提供时使用默认值"""

    value = tool_call.arguments.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value
