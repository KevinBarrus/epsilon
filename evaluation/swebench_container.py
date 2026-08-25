"""管理单个 SWE-bench 任务的 Agent 执行容器。"""

import asyncio
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from core.tools.command_executor import CommandExecution, terminate_process_group


CONTAINER_WORKDIR = "/testbed"


class SwebenchContainerError(RuntimeError):
    """表示 Docker 生命周期操作失败。"""


class SwebenchTaskContainer:
    """使用官方实例镜像承载单个 Agent 评测任务。"""

    def __init__(self, image: str, workspace: Path) -> None:
        """保存已由评测任务元数据提供的镜像和工作区。"""

        if not image:
            raise ValueError("container image must not be empty")
        self._image = image
        self._workspace = workspace.resolve()
        self._container_id: str | None = None

    @property
    def container_id(self) -> str | None:
        """返回已启动容器的标识。"""

        return self._container_id

    @property
    def workspace(self) -> Path:
        """返回与容器共享的评测工作区。"""

        return self._workspace

    async def start(self) -> str:
        """启动挂载评测工作区的常驻任务容器。"""

        if self._container_id is not None:
            raise RuntimeError("SWE-bench task container is already running")
        if not self._workspace.is_dir():
            raise SwebenchContainerError(f"评测工作区不存在：{self._workspace}")

        completed = await _run_docker(
            [
                "run",
                "--detach",
                "--user",
                _host_workspace_user(),
                "--network",
                "none",
                "--volume",
                f"{self._workspace}:{CONTAINER_WORKDIR}",
                "--workdir",
                CONTAINER_WORKDIR,
                self._image,
                "tail",
                "-f",
                "/dev/null",
            ]
        )
        if completed.returncode != 0:
            raise SwebenchContainerError(_docker_error("启动", completed))
        container_id = completed.stdout.strip()
        if not container_id:
            raise SwebenchContainerError("启动评测容器失败：Docker 未返回容器标识")
        self._container_id = container_id
        return container_id

    async def close(self) -> None:
        """强制删除任务容器，重复调用安全。"""

        container_id, self._container_id = self._container_id, None
        if container_id is None:
            return
        completed = await _run_docker(["rm", "--force", container_id])
        if completed.returncode != 0:
            raise SwebenchContainerError(_docker_error("清理", completed))

    @asynccontextmanager
    async def running(self) -> AsyncIterator[str]:
        """在作用域结束时回收容器。"""

        container_id = await self.start()
        try:
            yield container_id
        finally:
            await self.close()


class SwebenchContainerExecutor:
    """在已启动的 SWE-bench 任务容器中执行 Agent 命令。"""

    def __init__(self, container: SwebenchTaskContainer) -> None:
        """绑定单个评测任务容器。"""

        self._container = container

    async def execute(
        self,
        command: str,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandExecution:
        """在容器工作区运行命令，超时或取消时销毁容器。"""

        container_id = self._container.container_id
        if container_id is None:
            raise SwebenchContainerError("评测容器尚未启动")
        if cwd.resolve() != self._container.workspace:
            raise SwebenchContainerError("容器命令只能在评测工作区根目录执行")
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "--workdir",
            CONTAINER_WORKDIR,
            container_id,
            "sh",
            "-lc",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds
            )
        except (TimeoutError, asyncio.CancelledError):
            await terminate_process_group(process)
            await self._container.close()
            raise
        return CommandExecution(stdout, stderr, process.returncode)


async def _run_docker(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """在线程中执行短生命周期 Docker 管理命令。"""

    try:
        return await asyncio.to_thread(
            subprocess.run,
            ["docker", *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise SwebenchContainerError(f"Docker 不可用：{exc.strerror or exc}") from exc


def _docker_error(operation: str, completed: subprocess.CompletedProcess[str]) -> str:
    """生成不含调用参数的容器操作错误信息。"""

    detail = completed.stderr.strip() or completed.stdout.strip() or "未知 Docker 错误"
    return f"{operation}评测容器失败，退出码 {completed.returncode}：{detail}"


def _host_workspace_user() -> str:
    """返回可写入宿主绑定工作区的有效用户身份。"""

    return f"{os.geteuid()}:{os.getegid()}"
