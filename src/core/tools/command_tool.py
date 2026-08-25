"""实现命令执行工具。"""

import asyncio
from pathlib import Path

from ..model import ToolCall, ToolResult
from .args import string_argument
from .command_executor import CommandExecutor, HostCommandExecutor
from .output_limits import limit_tool_output
from .types import ToolDefinition, ToolHandler

COMMAND_TIMEOUT_SECONDS = 60.0


def create_run_command_tool(
    workspace: Path,
    timeout_seconds: float = COMMAND_TIMEOUT_SECONDS,
    executor: CommandExecutor | None = None,
) -> tuple[ToolDefinition, ToolHandler]:
    """创建以工作区为当前目录的命令执行工具。"""

    if timeout_seconds <= 0:
        raise ValueError("command timeout must be > 0")
    command_executor = executor or HostCommandExecutor()

    async def run_command(tool_call: ToolCall) -> ToolResult:
        command = string_argument(tool_call, "command")
        try:
            execution = await command_executor.execute(
                command,
                workspace,
                timeout_seconds,
            )
        except TimeoutError:
            return ToolResult(
                call_id=tool_call.call_id,
                content=f"command timed out after {timeout_seconds:g}s",
                is_error=True,
                error_category="tool_execution",
            )
        output = _format_output(execution.stdout, execution.stderr)
        if execution.returncode:
            output = f"exit code: {execution.returncode}\n{output}"
        output = limit_tool_output(output)
        return ToolResult(
            call_id=tool_call.call_id,
            content=output,
            is_error=execution.returncode != 0,
        )

    return (
        ToolDefinition(
            name="run_command",
            description="Execute a shell command in the current workspace",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            source="local",
            permission="command",
            idempotent=False,
        ),
        run_command,
    )


def _format_output(stdout: bytes, stderr: bytes) -> str:
    """合并命令的标准输出和错误输出。"""

    parts: list[str] = []
    if stdout:
        parts.append(stdout.decode(errors="replace").rstrip())
    if stderr:
        parts.append(f"stderr:\n{stderr.decode(errors='replace').rstrip()}")
    return "\n".join(parts) or "command executed successfully"
