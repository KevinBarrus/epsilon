import asyncio
import shlex
import sys
from pathlib import Path

import pytest

from core.tools import command_executor
from core.model import ToolCall
from core.tools import (
    ApprovalDecision,
    ApprovalResult,
    MAX_TOOL_OUTPUT_BYTES,
    PermissionManager,
    TRUNCATION_NOTICE,
    ToolManager,
    CommandExecution,
    create_edit_file_tool,
    create_run_command_tool,
    create_write_file_tool,
)


def _call(name: str, arguments: dict[str, object]) -> ToolCall:
    """构造测试用工具调用。"""

    return ToolCall(call_id="call-1", name=name, arguments=arguments)


def _manager(*tools: tuple) -> ToolManager:
    """注册指定的本地工具。"""

    async def approve(definition, tool_call, allow_session) -> ApprovalResult:
        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    manager = ToolManager(permission_manager=PermissionManager(approve))
    for definition, handler in tools:
        manager.register_local(definition, handler)
    return manager


@pytest.mark.asyncio
async def test_write_file_is_idempotent(tmp_path: Path) -> None:
    """测试写入相同内容时不会重复改变文件。"""

    manager = _manager(create_write_file_tool(tmp_path))
    call = _call("write_file", {"path": "src/a.txt", "content": "内容"})

    first = await manager.execute(call)
    second = await manager.execute(call)

    assert first.content.startswith("file written")
    assert second.content == "file content already matches the target"
    assert (tmp_path / "src/a.txt").read_text(encoding="utf-8") == "内容"


@pytest.mark.asyncio
async def test_write_file_allows_empty_content(tmp_path: Path) -> None:
    """测试写入工具允许创建空文件。"""

    manager = _manager(create_write_file_tool(tmp_path))

    result = await manager.execute(
        _call("write_file", {"path": "empty.txt", "content": ""})
    )

    assert result.is_error is False
    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_edit_file_requires_expected_old_content(tmp_path: Path) -> None:
    """测试编辑工具能正常修改、识别重复调用并拒绝状态冲突。"""

    path = tmp_path / "a.txt"
    path.write_text("旧内容", encoding="utf-8")
    manager = _manager(create_edit_file_tool(tmp_path))
    call = _call(
        "edit_file",
        {"path": "a.txt", "old_content": "旧内容", "new_content": "新内容"},
    )

    assert (await manager.execute(call)).content.startswith("file edited")
    assert (await manager.execute(call)).content == "file content already matches the target"

    path.write_text("外部修改", encoding="utf-8")
    conflict = await manager.execute(call)
    assert conflict.is_error is True
    assert "refusing to overwrite" in conflict.content


@pytest.mark.asyncio
async def test_edit_file_replaces_substring_preserving_surrounding_content(
    tmp_path: Path,
) -> None:
    """测试编辑工具按子串替换并保留其余内容（如末尾换行）。"""

    path = tmp_path / "a.txt"
    path.write_text("before\n", encoding="utf-8")
    manager = _manager(create_edit_file_tool(tmp_path))

    result = await manager.execute(
        _call(
            "edit_file",
            {"path": "a.txt", "old_content": "before", "new_content": "after"},
        )
    )

    assert result.is_error is False
    assert path.read_text(encoding="utf-8") == "after\n"


@pytest.mark.asyncio
async def test_edit_file_result_contains_diff(tmp_path: Path) -> None:
    """测试编辑结果附带 - 删除行与 + 新增行。"""

    path = tmp_path / "a.txt"
    path.write_text("old\nkeep\n", encoding="utf-8")
    manager = _manager(create_edit_file_tool(tmp_path))

    result = await manager.execute(
        _call(
            "edit_file",
            {"path": "a.txt", "old_content": "old", "new_content": "new"},
        )
    )

    content = result.content
    assert content.startswith("file edited")
    assert "-old" in content
    assert "+new" in content


@pytest.mark.asyncio
async def test_write_file_new_result_lists_added_lines(tmp_path: Path) -> None:
    """测试新建文件结果附带全部新增行，超长截断。"""

    manager = _manager(create_write_file_tool(tmp_path))

    result = await manager.execute(
        _call("write_file", {"path": "a.txt", "content": "1\n2\n3\n"})
    )

    content = result.content
    assert content.startswith("file written")
    assert "+1" in content and "+3" in content


@pytest.mark.asyncio
async def test_run_command_returns_stdout_and_exit_error(tmp_path: Path) -> None:
    """测试命令工具返回标准输出和非零退出错误。"""

    manager = _manager(create_run_command_tool(tmp_path))
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote('print(\"完成\")')}"
    result = await manager.execute(_call("run_command", {"command": command}))

    assert result.is_error is False
    assert result.content == "完成"
    assert (tmp_path / "created.txt").exists() is False


@pytest.mark.asyncio
async def test_run_command_marks_nonzero_exit_as_error(tmp_path: Path) -> None:
    """测试命令非零退出码会标记为错误。"""

    manager = _manager(create_run_command_tool(tmp_path))
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import sys; sys.exit(2)')}"

    result = await manager.execute(_call("run_command", {"command": command}))

    assert result.is_error is True
    assert "exit code: 2" in result.content


@pytest.mark.asyncio
async def test_run_command_truncates_large_output(tmp_path: Path) -> None:
    """测试命令工具会限制返回给模型的输出。"""

    manager = _manager(create_run_command_tool(tmp_path))
    script = f"print('x' * {MAX_TOOL_OUTPUT_BYTES + 1})"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    result = await manager.execute(_call("run_command", {"command": command}))

    assert result.content.endswith(TRUNCATION_NOTICE)
    assert len(result.content.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES


@pytest.mark.asyncio
async def test_run_command_returns_timeout_error(tmp_path: Path) -> None:
    """测试命令超时会返回结构化工具错误。"""

    manager = _manager(create_run_command_tool(tmp_path, timeout_seconds=0.1))
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(1)')}"

    result = await manager.execute(_call("run_command", {"command": command}))

    assert result.is_error is True
    assert result.error_category == "tool_execution"
    assert "timed out" in result.content


@pytest.mark.asyncio
async def test_run_command_uses_injected_executor(tmp_path: Path) -> None:
    """测试命令工具将执行参数交给注入的后端。"""

    class FakeExecutor:
        def __init__(self) -> None:
            self.arguments: tuple[str, Path, float] | None = None

        async def execute(
            self,
            command: str,
            cwd: Path,
            timeout_seconds: float,
        ) -> CommandExecution:
            self.arguments = (command, cwd, timeout_seconds)
            return CommandExecution(b"from fake", b"", 0)

    executor = FakeExecutor()
    manager = _manager(
        create_run_command_tool(tmp_path, timeout_seconds=12, executor=executor)
    )

    result = await manager.execute(_call("run_command", {"command": "check"}))

    assert result.content == "from fake"
    assert executor.arguments == ("check", tmp_path, 12)


@pytest.mark.asyncio
async def test_run_command_cancellation_stops_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试取消命令时会终止独立的子进程组。"""

    class HangingProcess:
        pid = 123
        returncode: int | None = None

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.communicate_calls = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_calls += 1
            if self.returncode is not None:
                return b"", b""
            self.started.set()
            await asyncio.Event().wait()
            return b"", b""

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    process = HangingProcess()
    kwargs: dict[str, object] = {}
    signals: list[int] = []

    async def create_process(*args, **received_kwargs):
        kwargs.update(received_kwargs)
        return process

    monkeypatch.setattr(command_executor.asyncio, "create_subprocess_shell", create_process)
    monkeypatch.setattr(
        command_executor.os,
        "killpg",
        lambda process_id, value: signals.append(value),
    )
    _, handler = create_run_command_tool(tmp_path)
    task = asyncio.create_task(handler(_call("run_command", {"command": "sleep 10"})))
    await process.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert kwargs["start_new_session"] is True
    assert signals == [command_executor.signal.SIGTERM]
    assert process.communicate_calls == 2
