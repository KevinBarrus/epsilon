"""验证 Worker 独立 Git worktree 的创建、合并与冲突保全。"""

import subprocess
from pathlib import Path

import pytest

from core.worktree import (
    WorktreeError, commit_worktree, create_worktree, merge_branch,
    remove_worktree,
)


def git(repo: Path, *args: str) -> str:
    """只在测试临时仓库中执行 Git 命令。"""
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """建立独立基线仓库，不接触项目自身的 Git 历史。"""
    root = tmp_path / "main"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.name", "Worktree Test")
    git(root, "config", "user.email", "worktree@example.invalid")
    (root / "shared.txt").write_text("baseline\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-m", "baseline")
    return root


def test_worktree_roundtrip_merges_worker_change(repo: Path) -> None:
    """Worker 的提交合入主副本后，隔离目录可清理。"""
    worktree = create_worktree(repo, "worker-1")
    assert worktree.parent == repo.parent
    (worktree / "new.txt").write_text("from worker\n", encoding="utf-8")
    assert not (repo / "new.txt").exists()

    branch = commit_worktree(worktree)
    result = merge_branch(repo, branch)
    remove_worktree(repo, worktree)

    assert result.status == "merged"
    assert (repo / "new.txt").read_text(encoding="utf-8") == "from worker\n"
    assert not worktree.exists()


def test_worktree_removal_after_ignored_dependency_install(repo: Path) -> None:
    """Worker 安装依赖后仍能清理隔离目录。"""
    with (repo / ".git/info/exclude").open("a", encoding="utf-8") as stream:
        stream.write("\nnode_modules/\n")
    worktree = create_worktree(repo, "dependencies")
    dependency = worktree / "node_modules/example/index.js"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("module.exports = 1", encoding="utf-8")

    branch = commit_worktree(worktree)
    assert merge_branch(repo, branch).status == "merged"
    remove_worktree(repo, worktree)

    assert not worktree.exists()


def test_conflict_preserves_worker_branch(repo: Path) -> None:
    """两名 Worker 改同一文件时，失败分支仍能找回内容。"""
    first = create_worktree(repo, "first")
    second = create_worktree(repo, "second")
    (first / "shared.txt").write_text("first\n", encoding="utf-8")
    (second / "shared.txt").write_text("second\n", encoding="utf-8")
    first_branch = commit_worktree(first)
    second_branch = commit_worktree(second)

    assert merge_branch(repo, first_branch).status == "merged"
    conflict = merge_branch(repo, second_branch)
    remove_worktree(repo, first)
    remove_worktree(repo, second)

    assert conflict.status == "conflict"
    assert "shared.txt" in conflict.conflicted_paths
    assert (repo / "shared.txt").read_text(encoding="utf-8") == "first\n"
    assert git(repo, "show", f"{second_branch}:shared.txt") == "second"
    assert git(repo, "status", "--porcelain") == ""


def test_merge_includes_worker_commits_made_before_final_commit(repo: Path) -> None:
    """Worker 若自行提交过，也不能只合并分支最后一个提交。"""
    worktree = create_worktree(repo, "multiple")
    (worktree / "first.txt").write_text("first", encoding="utf-8")
    git(worktree, "add", "-A")
    git(worktree, "commit", "-m", "first worker commit")
    (worktree / "second.txt").write_text("second", encoding="utf-8")
    branch = commit_worktree(worktree)

    assert merge_branch(repo, branch).status == "merged"
    assert (repo / "first.txt").read_text(encoding="utf-8") == "first"
    assert (repo / "second.txt").read_text(encoding="utf-8") == "second"
    remove_worktree(repo, worktree)


def test_multi_commit_conflict_rolls_back_entire_worker(repo: Path) -> None:
    """后一个提交冲突时，前一个提交也不能半合入主工作区。"""
    first = create_worktree(repo, "winner")
    second = create_worktree(repo, "loser")
    (first / "shared.txt").write_text("winner", encoding="utf-8")
    winner_branch = commit_worktree(first)
    (second / "independent.txt").write_text("preserved", encoding="utf-8")
    git(second, "add", "-A")
    git(second, "commit", "-m", "first loser commit")
    (second / "shared.txt").write_text("loser", encoding="utf-8")
    loser_branch = commit_worktree(second)

    assert merge_branch(repo, winner_branch).status == "merged"
    assert merge_branch(repo, loser_branch).status == "conflict"
    assert not (repo / "independent.txt").exists()
    assert git(repo, "show", f"{loser_branch}:independent.txt") == "preserved"
    assert git(repo, "status", "--porcelain") == ""
    remove_worktree(repo, first)
    remove_worktree(repo, second)


def test_worktree_requires_clean_git_checkout(tmp_path: Path, repo: Path) -> None:
    """非仓库和脏工作区明确报错，不隐式提交用户文件。"""
    with pytest.raises(WorktreeError, match="Git checkout"):
        create_worktree(tmp_path / "not-a-repo", "worker")
    (repo / "uncommitted.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(WorktreeError, match="uncommitted"):
        create_worktree(repo, "worker")
    assert (repo / "uncommitted.txt").read_text(encoding="utf-8") == "user data"
