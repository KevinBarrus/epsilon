"""验证 SWE-bench 任务容器的生命周期。"""

import asyncio
import subprocess
from pathlib import Path

import pytest

from core.tools import command_executor
from evaluation.swebench_container import (
    CONTAINER_WORKDIR,
    SwebenchContainerExecutor,
    SwebenchContainerError,
    SwebenchTaskContainer,
)


@pytest.mark.asyncio
async def test_task_container_uses_isolated_official_image_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试容器以官方镜像和固定工作目录启动。"""

    calls: list[list[str]] = []

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "container-1\n", "")

    monkeypatch.setattr("evaluation.swebench_container.subprocess.run", run)
    container = SwebenchTaskContainer("swebench/example:latest", tmp_path)

    container_id = await container.start()

    assert container_id == "container-1"
    assert calls == [
        [
            "docker",
            "run",
            "--detach",
            "--user",
            "root",
            "--network",
            "none",
            "--volume",
            f"{tmp_path.resolve()}:{CONTAINER_WORKDIR}",
            "--workdir",
            CONTAINER_WORKDIR,
            "swebench/example:latest",
            "tail",
            "-f",
            "/dev/null",
        ]
    ]


@pytest.mark.asyncio
async def test_task_container_removes_container_after_context_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试评测异常后仍会强制回收任务容器。"""

    calls: list[list[str]] = []

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        stdout = "container-1\n" if command[1] == "run" else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr("evaluation.swebench_container.subprocess.run", run)
    container = SwebenchTaskContainer("swebench/example:latest", tmp_path)

    with pytest.raises(RuntimeError, match="agent failed"):
        async with container.running():
            raise RuntimeError("agent failed")

    assert container.container_id is None
    assert calls[-1] == ["docker", "rm", "--force", "container-1"]


@pytest.mark.asyncio
async def test_task_container_reports_docker_start_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试 Docker 启动错误会保留评测环境分类。"""

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 125, "", "image missing")

    monkeypatch.setattr("evaluation.swebench_container.subprocess.run", run)

    with pytest.raises(SwebenchContainerError, match="启动评测容器失败"):
        await SwebenchTaskContainer("swebench/example:latest", tmp_path).start()


@pytest.mark.asyncio
async def test_task_container_reports_missing_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试 Docker 缺失会成为可归因的评测环境错误。"""

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("docker")

    monkeypatch.setattr("evaluation.swebench_container.subprocess.run", run)

    with pytest.raises(SwebenchContainerError, match="Docker 不可用"):
        await SwebenchTaskContainer("swebench/example:latest", tmp_path).start()


@pytest.mark.asyncio
async def test_container_executor_runs_command_in_task_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试容器命令固定运行在共享的任务工作区。"""

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "container-1\n", "")

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"test output", b""

    command: tuple[str, ...] | None = None
    kwargs: dict[str, object] = {}

    async def create_process(*arguments: str, **received_kwargs) -> Process:
        nonlocal command
        command = arguments
        kwargs.update(received_kwargs)
        return Process()

    monkeypatch.setattr("evaluation.swebench_container.subprocess.run", run)
    monkeypatch.setattr(
        "evaluation.swebench_container.asyncio.create_subprocess_exec", create_process
    )
    container = SwebenchTaskContainer("swebench/example:latest", tmp_path)
    await container.start()

    result = await SwebenchContainerExecutor(container).execute("pytest -q", tmp_path, 30)

    assert result.stdout == b"test output"
    assert command == (
        "docker",
        "exec",
        "--workdir",
        CONTAINER_WORKDIR,
        "container-1",
        "sh",
        "-lc",
        "pytest -q",
    )
    assert kwargs["start_new_session"] is True


@pytest.mark.asyncio
async def test_container_executor_timeout_removes_task_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试命令超时后会回收进程组与整个任务容器。"""

    docker_calls: list[list[str]] = []
    signals: list[int] = []

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        docker_calls.append(command)
        stdout = "container-1\n" if command[1] == "run" else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    class HangingProcess:
        pid = 123
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            if self.returncode is not None:
                return b"", b""
            await asyncio.Event().wait()
            return b"", b""

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    async def create_process(*arguments: str, **kwargs) -> HangingProcess:
        return HangingProcess()

    monkeypatch.setattr("evaluation.swebench_container.subprocess.run", run)
    monkeypatch.setattr(
        "evaluation.swebench_container.asyncio.create_subprocess_exec", create_process
    )
    monkeypatch.setattr(
        command_executor.os,
        "killpg",
        lambda process_id, value: signals.append(value),
    )
    container = SwebenchTaskContainer("swebench/example:latest", tmp_path)
    await container.start()

    with pytest.raises(TimeoutError):
        await SwebenchContainerExecutor(container).execute("sleep 10", tmp_path, 0.01)

    assert signals == [command_executor.signal.SIGTERM]
    assert container.container_id is None
    assert docker_calls[-1] == ["docker", "rm", "--force", "container-1"]


@pytest.mark.asyncio
async def test_container_executor_cancellation_removes_task_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试取消命令后仍会回收任务容器。"""

    docker_calls: list[list[str]] = []

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        docker_calls.append(command)
        stdout = "container-1\n" if command[1] == "run" else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    class HangingProcess:
        pid = 123
        returncode: int | None = None

        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def communicate(self) -> tuple[bytes, bytes]:
            if self.returncode is not None:
                return b"", b""
            self.started.set()
            await asyncio.Event().wait()
            return b"", b""

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    process = HangingProcess()

    async def create_process(*arguments: str, **kwargs) -> HangingProcess:
        return process

    monkeypatch.setattr("evaluation.swebench_container.subprocess.run", run)
    monkeypatch.setattr(
        "evaluation.swebench_container.asyncio.create_subprocess_exec", create_process
    )
    monkeypatch.setattr(command_executor.os, "killpg", lambda *args: None)
    container = SwebenchTaskContainer("swebench/example:latest", tmp_path)
    await container.start()
    task = asyncio.create_task(
        SwebenchContainerExecutor(container).execute("sleep 10", tmp_path, 30)
    )
    await process.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert container.container_id is None
    assert docker_calls[-1] == ["docker", "rm", "--force", "container-1"]
