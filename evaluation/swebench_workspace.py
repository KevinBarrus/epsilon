"""准备不暴露仓库历史的 SWE-bench 评测工作区"""

import io
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path


class EvaluationWorkspaceError(RuntimeError):
    """评测工作区无法安全创建时抛出"""


@dataclass(frozen=True)
class EvaluationWorkspace:
    """保存单次评测的 Agent 工作区与外部会话目录"""

    workspace: Path
    session_root: Path


def prepare_evaluation_workspace(
    repository: Path,
    base_commit: str,
    result_root: Path,
) -> EvaluationWorkspace:
    """从基线提交导出无 Git 元数据的独立评测工作区"""

    archive = _archive_commit(repository, base_commit)
    workspaces_root = result_root / "workspaces"
    workspaces_root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="swebench-", dir=workspaces_root))

    try:
        _extract_archive(archive, workspace)
        _assert_clean_workspace(workspace)
    except Exception:
        _remove_empty_workspace(workspace)
        raise

    session_root = result_root / "sessions" / workspace.name
    session_root.mkdir(parents=True, exist_ok=True)
    return EvaluationWorkspace(workspace=workspace, session_root=session_root)


def _archive_commit(repository: Path, base_commit: str) -> bytes:
    """调用 Git 导出指定提交的受版本控制文件"""

    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "archive", "--format=tar", base_commit],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise EvaluationWorkspaceError("无法启动 Git 导出评测基线") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise EvaluationWorkspaceError(f"无法导出基线提交 {base_commit}: {message}") from exc
    return result.stdout


def _extract_archive(archive: bytes, destination: Path) -> None:
    """安全解压 Git archive，拒绝任何越界成员"""

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar.getmembers():
            path = (destination / member.name).resolve()
            try:
                path.relative_to(destination.resolve())
            except ValueError as exc:
                raise EvaluationWorkspaceError("基线归档包含越界路径") from exc
        tar.extractall(destination, filter="data")


def _assert_clean_workspace(workspace: Path) -> None:
    """确认 Agent 输入目录不包含可读取的仓库历史"""

    if (workspace / ".git").exists():
        raise EvaluationWorkspaceError("评测工作区不得包含 .git")


def _remove_empty_workspace(workspace: Path) -> None:
    """创建失败时仅删除尚未成功交付的临时工作区"""

    for path in sorted(workspace.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        else:
            path.rmdir()
    workspace.rmdir()
