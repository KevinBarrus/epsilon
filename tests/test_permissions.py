import pytest

from core.model import ToolCall
from core.tools import (
    ApprovalDecision,
    ApprovalResult,
    PermissionDenied,
    PermissionManager,
    ToolDefinition,
)


def _definition(permission: str, *, idempotent: bool | None = None) -> ToolDefinition:
    """构造测试用工具定义。"""

    return ToolDefinition(
        name="test_tool",
        description="测试工具",
        parameters={"type": "object"},
        source="local",
        permission=permission,  # type: ignore[arg-type]
        idempotent=permission == "read" if idempotent is None else idempotent,
    )


def _call() -> ToolCall:
    """构造测试用工具调用。"""

    return ToolCall(call_id="call-1", name="test_tool", arguments={})


@pytest.mark.asyncio
async def test_read_tool_is_allowed_without_confirmation() -> None:
    """测试只读工具不需要外部确认。"""

    await PermissionManager().authorize(_definition("read"), _call())


@pytest.mark.asyncio
async def test_mutating_tool_is_denied_without_confirmation() -> None:
    """测试没有确认回调时拒绝有副作用的工具。"""

    with pytest.raises(PermissionDenied, match="requires user approval"):
        await PermissionManager().authorize(_definition("write"), _call())


@pytest.mark.asyncio
async def test_confirmation_callback_receives_tool_context() -> None:
    """测试确认回调可以获取工具定义和调用参数。"""

    received: list[tuple[str, str, str]] = []

    async def approve(
        definition: ToolDefinition,
        tool_call: ToolCall,
        allow_session: bool,
    ) -> ApprovalResult:
        received.append((definition.name, tool_call.call_id, str(allow_session)))
        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    result = await PermissionManager(approve).authorize(_definition("command"), _call())

    assert received == [("test_tool", "call-1", "False")]
    assert result.decision == ApprovalDecision.ALLOW_ONCE


@pytest.mark.asyncio
async def test_rejected_confirmation_returns_denial() -> None:
    """测试确认回调拒绝时返回拒绝结果。"""

    async def reject(
        definition: ToolDefinition,
        tool_call: ToolCall,
        allow_session: bool,
    ) -> ApprovalResult:
        return ApprovalResult(
            ApprovalDecision.DENY,
            feedback="先只读取文件",
        )

    result = await PermissionManager(reject).authorize(_definition("write"), _call())

    assert result == ApprovalResult(ApprovalDecision.DENY, "先只读取文件")


@pytest.mark.asyncio
async def test_session_grant_skips_future_confirmation() -> None:
    """测试当前 Session 授权后不再重复确认同一个工具。"""

    calls = 0

    async def approve(
        definition: ToolDefinition,
        tool_call: ToolCall,
        allow_session: bool,
    ) -> ApprovalResult:
        nonlocal calls
        calls += 1
        return ApprovalResult(ApprovalDecision.ALLOW_SESSION)

    manager = PermissionManager(approve)
    first = await manager.authorize(_definition("write", idempotent=True), _call())
    second = await manager.authorize(_definition("write", idempotent=True), _call())

    assert first.decision == ApprovalDecision.ALLOW_SESSION
    assert second.decision == ApprovalDecision.ALLOW_SESSION
    assert calls == 1


@pytest.mark.asyncio
async def test_command_never_grants_session_permission() -> None:
    """测试命令工具即使请求会话授权也只会单次放行。"""

    calls = 0

    async def approve(definition, tool_call, allow_session) -> ApprovalResult:
        nonlocal calls
        calls += 1
        assert allow_session is False
        return ApprovalResult(ApprovalDecision.ALLOW_SESSION)

    manager = PermissionManager(approve)
    first = await manager.authorize(_definition("command"), _call())
    second = await manager.authorize(_definition("command"), _call())

    assert first.decision == ApprovalDecision.ALLOW_ONCE
    assert second.decision == ApprovalDecision.ALLOW_ONCE
    assert calls == 2


@pytest.mark.asyncio
async def test_non_idempotent_tool_never_grants_session_permission() -> None:
    """测试不可幂等工具的会话授权会退化为单次授权。"""

    calls = 0

    async def approve(definition, tool_call, allow_session) -> ApprovalResult:
        nonlocal calls
        calls += 1
        assert allow_session is False
        return ApprovalResult(ApprovalDecision.ALLOW_SESSION)

    manager = PermissionManager(approve)
    first = await manager.authorize(_definition("write", idempotent=False), _call())
    second = await manager.authorize(_definition("write", idempotent=False), _call())

    assert first.decision == ApprovalDecision.ALLOW_ONCE
    assert second.decision == ApprovalDecision.ALLOW_ONCE
    assert calls == 2
