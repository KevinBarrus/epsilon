"""用 Git worktree 隔离 Worker 写入，并串行合并结果。"""

import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4


class WorktreeError(RuntimeError):
    """表示隔离仓库不满足条件或 Git 操作失败。"""


@dataclass(frozen=True)
class MergeResult:
    """记录一次 Worker 分支合并结果；冲突分支始终保留。"""

    branch: str
    status: Literal["merged", "conflict"]
    conflicted_paths: tuple[str, ...] = ()


# ponytail: 同进程所有仓库共用一把锁；若多仓库吞吐成为瓶颈，再按仓库分锁。
_GIT_LOCK = threading.RLock()


def _git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """以参数列表执行 Git，避免任务名称进入 shell。"""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorktreeError(f"Git command failed: {exc}") from exc
    if check and result.returncode:
        raise WorktreeError(result.stderr.strip() or result.stdout.strip() or "Git command failed")
    return result


def _checkout_root(path: Path) -> Path:
    """要求传入 Git 检出根目录，不接受普通目录或仓库子目录。"""
    root = path.resolve()
    result = _git(root, "rev-parse", "--show-toplevel", check=False)
    if result.returncode or Path(result.stdout.strip()).resolve() != root:
        raise WorktreeError(f"worktree isolation requires a Git checkout: {root}")
    return root


def _require_clean(root: Path) -> None:
    """拒绝覆盖主工作区未提交内容，也不替用户隐式提交。"""
    if _git(root, "status", "--porcelain", "--untracked-files=all").stdout.strip():
        raise WorktreeError("worktree isolation requires no uncommitted changes")


def create_worktree(repo_root: Path, task_id: str) -> Path:
    """从当前 HEAD 为 Worker 建立同盘兄弟工作区。"""
    with _GIT_LOCK:
        root = _checkout_root(repo_root)
        _require_clean(root)
        slug = re.sub(r"[^A-Za-z0-9_-]", "-", task_id).strip("-")[:24] or "task"
        suffix = uuid4().hex[:12]
        branch = f"epsilon/worker-{slug}-{suffix}"
        path = root.parent / f".{root.name}-worker-{slug}-{suffix}"
        _git(root, "worktree", "add", "-b", branch, str(path), "HEAD")
        return path


def commit_worktree(path: Path, message: str = "feat(worker): 固化隔离工作区改动") -> str:
    """固化 Worker 改动；没有改动时保留基线分支。"""
    with _GIT_LOCK:
        root = _checkout_root(path)
        branch = _git(root, "branch", "--show-current").stdout.strip()
        if not branch:
            raise WorktreeError("worktree must be on a branch")
        _git(root, "add", "-A")
        if _git(root, "diff", "--cached", "--quiet", check=False).returncode:
            _git(root, "commit", "-m", message)
        return branch


def merge_branch(repo_root: Path, branch: str) -> MergeResult:
    """串行 cherry-pick；冲突时恢复主工作区并留下 Worker 分支。"""
    with _GIT_LOCK:
        root = _checkout_root(repo_root)
        _require_clean(root)
        if not branch.startswith("epsilon/worker-"):
            raise WorktreeError("refusing to merge an unrelated branch")
        _git(root, "rev-parse", "--verify", branch)
        if _git(root, "merge-base", "--is-ancestor", branch, "HEAD", check=False).returncode == 0:
            return MergeResult(branch, "merged")
        base = _git(root, "merge-base", "HEAD", branch).stdout.strip()
        commits = _git(root, "rev-list", "--reverse", f"{base}..{branch}").stdout.splitlines()
        if not commits:
            return MergeResult(branch, "merged")
        result = _git(root, "cherry-pick", *commits, check=False)
        if result.returncode == 0:
            return MergeResult(branch, "merged")
        paths = tuple(
            _git(root, "diff", "--name-only", "--diff-filter=U").stdout.splitlines()
        )
        if _git(root, "rev-parse", "--verify", "CHERRY_PICK_HEAD", check=False).returncode == 0:
            _git(root, "cherry-pick", "--abort")
        if paths:
            return MergeResult(branch, "conflict", paths)
        raise WorktreeError(result.stderr.strip() or result.stdout.strip() or "cherry-pick failed")


def remove_worktree(repo_root: Path, path: Path) -> None:
    """只清理本模块创建的兄弟工作区；失败时保留文件供恢复。"""
    with _GIT_LOCK:
        root = _checkout_root(repo_root)
        target = path.resolve()
        if target.parent != root.parent or not target.name.startswith(f".{root.name}-worker-"):
            raise WorktreeError("refusing to remove an unrelated worktree")
        registered = _git(root, "worktree", "list", "--porcelain").stdout
        if f"worktree {target}\n" not in registered:
            raise WorktreeError("worktree is not registered in this repository")
        _git(root, "worktree", "remove", str(target))
