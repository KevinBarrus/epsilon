"""校验模型返回的工具参数。"""

from collections.abc import Mapping

from ..model import ToolCall
from .types import ToolDefinition


class ToolArgumentError(ValueError):
    """工具参数不符合定义时抛出的异常。"""


def validate_tool_arguments(
    definition: ToolDefinition,
    tool_call: ToolCall,
) -> None:
    """根据工具定义的最小 JSON Schema 校验调用参数。"""

    schema = definition.parameters
    arguments = tool_call.arguments
    if not isinstance(arguments, dict):
        raise ToolArgumentError("tool arguments must be a JSON object")

    required = schema.get("required", [])
    if not isinstance(required, list):
        raise ToolArgumentError("tool required fields must be an array")
    for name in required:
        if not isinstance(name, str) or name not in arguments:
            raise ToolArgumentError(f"missing tool argument: {name}")

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ToolArgumentError("tool properties must be an object")
    for name, value in arguments.items():
        property_schema = properties.get(name)
        if property_schema is None:
            if definition.source == "local":
                raise ToolArgumentError(f"unknown tool argument: {name}")
            continue
        if not isinstance(property_schema, Mapping):
            raise ToolArgumentError(f"invalid argument definition: {name}")
        _validate_value(name, value, property_schema)


def _validate_value(name: str, value: object, schema: Mapping[object, object]) -> None:
    """校验单个参数的基础 JSON 类型。"""

    value_type = schema.get("type")
    if value_type == "string" and not isinstance(value, str):
        raise ToolArgumentError(f"argument {name} must be a string")
    if value_type == "object" and not isinstance(value, dict):
        raise ToolArgumentError(f"argument {name} must be an object")
    if value_type == "array" and not isinstance(value, list):
        raise ToolArgumentError(f"argument {name} must be an array")
    if value_type == "boolean" and not isinstance(value, bool):
        raise ToolArgumentError(f"argument {name} must be a boolean")
    if value_type == "integer" and (
        not isinstance(value, int) or isinstance(value, bool)
    ):
        raise ToolArgumentError(f"argument {name} must be an integer")
    if value_type == "number" and (
        not isinstance(value, int | float) or isinstance(value, bool)
    ):
        raise ToolArgumentError(f"argument {name} must be a number")
    minimum = schema.get("minimum")
    if isinstance(minimum, int | float) and isinstance(value, int | float) and not isinstance(value, bool):
        if value < minimum:
            raise ToolArgumentError(f"argument {name} must be >= {minimum}")
