"""定义工具管理层使用的统一类型。"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from ..model import ToolCall, ToolResult


ToolSource = Literal["local", "mcp"]
ToolPermission = Literal["read", "write", "command"]
ToolExecutionMode = Literal["parallel", "sequential"]


@dataclass(frozen=True)
class ToolRoute:
    """标识工具所属来源和具体提供者。"""

    source: ToolSource
    provider_id: str


@dataclass(frozen=True)
class ToolDefinition:
    """描述一个可以提供给模型的工具。"""

    name: str
    description: str
    parameters: dict[str, object]
    source: ToolSource
    permission: ToolPermission
    idempotent: bool
    execution_mode: ToolExecutionMode = "sequential"
    capability: str | None = None
    provider_id: str = "builtin"
    provider_tool_name: str | None = None

    @property
    def route(self) -> ToolRoute:
        """返回当前工具的完整路由身份。"""

        return ToolRoute(self.source, self.provider_id)


class ToolExecutor(Protocol):
    """定义工具执行器需要提供的最小接口。"""

    async def __call__(self, tool_call: ToolCall) -> ToolResult:
        """执行一次工具调用并返回结构化结果。"""


ToolHandler = Callable[[ToolCall], Awaitable[ToolResult]]
