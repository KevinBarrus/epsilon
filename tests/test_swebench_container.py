"""验证 SWE-bench 任务容器的生命周期。"""

import subprocess
from pathlib import Path

import pytest

from evaluation.swebench_container import (
    CONTAINER_WORKDIR,
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
