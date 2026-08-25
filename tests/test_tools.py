from collections.abc import Awaitable, Callable

import pytest

from core.model import ToolCall, ToolResult
from core.tools import (
    ApprovalDecision,
    ApprovalResult,
    LocalToolRegistry,
    PermissionManager,
    ToolDefinition,
    ToolManager,
)
from core.tools.registry import ToolRegistrationError


def _definition(
    name: str = "read_file",
    source: str = "local",
) -> ToolDefinition:
    """构造测试用工具定义。"""

    return ToolDefinition(
        name=name,
        description="读取文件",
        parameters={"type": "object"},
        source=source,  # type: ignore[arg-type]
        permission="read",
        idempotent=True,
    )


def _handler(content: str = "完成") -> Callable[[ToolCall], Awaitable[ToolResult]]:
    """构造返回固定结果的测试执行器。"""

    async def execute(tool_call: ToolCall) -> ToolResult:
        return ToolResult(call_id=tool_call.call_id, content=content)

    return execute


def test_local_registry_registers_and_lists_tools() -> None:
    """测试本地注册表可以注册、查找和列出工具。"""

    registry = LocalToolRegistry()
    definition = _definition()
    registry.register(definition, _handler())

    assert registry.get("read_file") is not None
    assert registry.definitions() == [definition]
    assert definition.execution_mode == "sequential"


def test_local_registry_rejects_duplicate_or_non_local_tools() -> None:
    """测试本地注册表拒绝重复工具和 MCP 工具。"""

    registry = LocalToolRegistry()
    registry.register(_definition(), _handler())

    with pytest.raises(ToolRegistrationError, match="tool already registered"):
        registry.register(_definition(), _handler())
    with pytest.raises(ToolRegistrationError, match="only accepts local tools"):
        registry.register(_definition("mcp_tool", "mcp"), _handler())


@pytest.mark.asyncio
async def test_tool_manager_executes_registered_tool() -> None:
    """测试工具管理器可以调度已注册工具。"""

    manager = ToolManager()
    manager.register_local(_definition(), _handler("文件内容"))

    result = await manager.execute(
        ToolCall(call_id="call-1", name="read_file", arguments={})
    )

    assert result == ToolResult(call_id="call-1", content="文件内容")


@pytest.mark.asyncio
async def test_tool_manager_returns_error_for_unknown_tool() -> None:
    """测试调用不存在的工具时返回结构化错误。"""

    result = await ToolManager().execute(
        ToolCall(call_id="call-1", name="missing", arguments={})
    )

    assert result.is_error is True
    assert result.call_id == "call-1"
    assert "tool not found" in result.content
    assert result.error_category == "invalid_request"


@pytest.mark.asyncio
async def test_tool_manager_converts_handler_error() -> None:
    """测试工具执行异常会转换为结构化错误。"""

    async def fail(tool_call: ToolCall) -> ToolResult:
        raise RuntimeError("读取失败")

    manager = ToolManager()
    manager.register_local(_definition(), fail)

    result = await manager.execute(
        ToolCall(call_id="call-1", name="read_file", arguments={})
    )

    assert result == ToolResult(
        call_id="call-1",
        content="tool execution failed, adjust based on the error",
        is_error=True,
        error_category="tool_execution",
    )


@pytest.mark.asyncio
async def test_tool_manager_denies_write_without_confirmation() -> None:
    """测试工具管理器不会执行未经确认的写工具。"""

    executed = False

    async def handler(tool_call: ToolCall) -> ToolResult:
        nonlocal executed
        executed = True
        return ToolResult(call_id=tool_call.call_id, content="不应执行")

    definition = _definition()
    definition = ToolDefinition(
        name=definition.name,
        description=definition.description,
        parameters=definition.parameters,
        source=definition.source,
        permission="write",
        idempotent=False,
    )
    manager = ToolManager()
    manager.register_local(definition, handler)

    result = await manager.execute(
        ToolCall(call_id="call-1", name="read_file", arguments={})
    )

    assert result.is_error is True
    assert "rejected" in result.content
    assert result.error_category == "tool_permission"
    assert executed is False


@pytest.mark.asyncio
async def test_tool_manager_uses_injected_approval() -> None:
    """测试工具管理器可以使用应用层注入的确认回调。"""

    async def approve(definition, tool_call, allow_session):
        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    manager = ToolManager(permission_manager=PermissionManager(approve))
    definition = ToolDefinition(
        name="write_file",
        description="写入文件",
        parameters={"type": "object"},
        source="local",
        permission="write",
        idempotent=True,
    )
    manager.register_local(definition, _handler("已写入"))

    result = await manager.execute(
        ToolCall(call_id="call-1", name="write_file", arguments={})
    )

    assert result == ToolResult(call_id="call-1", content="已写入")
