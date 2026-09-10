import json
import sys
from pathlib import Path

import pytest

from core.model import ToolCall
from core.tools import (
    ApprovalDecision,
    ApprovalResult,
    MAX_TOOL_OUTPUT_BYTES,
    PermissionManager,
    StdioMcpProvider,
    ToolManager,
)
from core.tools.output_limits import TRUNCATION_NOTICE


SERVER = r'''
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "notifications/initialized":
        continue
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "test"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "remote_echo", "description": "回显", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True, "idempotentHint": True}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": request["params"]["arguments"]["text"]}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
'''


@pytest.mark.asyncio
async def test_stdio_provider_lists_and_calls_tools(tmp_path: Path) -> None:
    """测试 stdio Provider 可以握手、发现和调用工具。"""

    provider = StdioMcpProvider(
        [sys.executable, "-u", "-c", SERVER],
        provider_id="test-server",
        cwd=tmp_path,
        trusted_read_tools=("remote_echo",),
    )

    definitions = await provider.list_tools()
    result = await provider.call_tool(
        ToolCall("call-1", "remote_echo", {"text": "hello"})
    )
    await provider.close()

    assert definitions[0].name == "remote_echo"
    assert definitions[0].provider_id == "test-server"
    assert definitions[0].permission == "read"
    assert definitions[0].idempotent is False
    assert result.content == "hello"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_stdio_provider_tools_are_available_through_tool_manager(
    tmp_path: Path,
) -> None:
    """测试发现的 stdio MCP 工具可通过统一工具入口执行。"""

    provider = StdioMcpProvider(
        [sys.executable, "-u", "-c", SERVER],
        provider_id="test-server",
        cwd=tmp_path,
        trusted_read_tools=("remote_echo",),
    )
    manager = ToolManager()
    await manager.register_mcp_provider(provider)
    result = await manager.execute(
        ToolCall("call-1", "mcp_test-server_remote_echo", {"text": "hello"})
    )
    await provider.close()

    assert result.content == "hello"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_stdio_provider_does_not_trust_server_permission_hints(
    tmp_path: Path,
) -> None:
    """测试远端自报只读和幂等不会绕过本地审批。"""

    approvals = 0

    async def approve(definition, tool_call, allow_session) -> ApprovalResult:
        nonlocal approvals
        approvals += 1
        assert definition.permission == "write"
        assert definition.idempotent is False
        assert allow_session is False
        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    provider = StdioMcpProvider(
        [sys.executable, "-u", "-c", SERVER],
        provider_id="test-server",
        cwd=tmp_path,
    )
    manager = ToolManager(permission_manager=PermissionManager(approve))
    await manager.register_mcp_provider(provider)
    result = await manager.execute(
        ToolCall("call-1", "mcp_test-server_remote_echo", {"text": "hello"})
    )
    await provider.close()

    assert result.is_error is False
    assert approvals == 1


@pytest.mark.asyncio
async def test_stdio_provider_limits_long_tool_output(tmp_path: Path) -> None:
    """测试 MCP 工具结果不能绕过统一输出上限。"""

    provider = StdioMcpProvider(
        [sys.executable, "-u", "-c", SERVER],
        provider_id="test-server",
        cwd=tmp_path,
    )
    result = await provider.call_tool(
        ToolCall("call-1", "remote_echo", {"text": "x" * (MAX_TOOL_OUTPUT_BYTES + 1)})
    )
    await provider.close()

    assert TRUNCATION_NOTICE in result.content
    assert len(result.content.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES


@pytest.mark.asyncio
async def test_stdio_provider_times_out_unresponsive_tool_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试 MCP 工具无响应时返回结构化错误并关闭失效进程。"""

    hanging_server = SERVER.replace(
        'elif method == "tools/call":\n        result = {"content": [{"type": "text", "text": request["params"]["arguments"]["text"]}]}',
        'elif method == "tools/call":\n        continue',
    )
    monkeypatch.setattr("core.tools.mcp_stdio.MCP_REQUEST_TIMEOUT_SECONDS", 0.05)
    provider = StdioMcpProvider(
        [sys.executable, "-u", "-c", hanging_server],
        provider_id="test-server",
        cwd=tmp_path,
    )
    await provider.list_tools()
    result = await provider.call_tool(ToolCall("call-1", "remote_echo", {}))

    assert result.is_error is True
    assert result.error_category == "tool_execution"
    assert "timed out" in result.content
