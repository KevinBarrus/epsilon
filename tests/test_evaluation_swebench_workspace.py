"""验证真实仓库评测工作区的隔离边界"""

import io
import tarfile
from pathlib import Path

import pytest

from evaluation.swebench_workspace import (
    EvaluationWorkspaceError,
    _assert_clean_workspace,
    _extract_archive,
    prepare_evaluation_workspace,
)


def _archive_with_file(name: str, content: str) -> bytes:
    """构造一个用于隔离测试的最小 Git archive 内容"""

    buffer = io.BytesIO()
    data = content.encode()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_prepare_workspace_exports_source_without_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试评测工作区只得到基线源码，会话目录位于工作区外"""

    archive = _archive_with_file("package/module.py", "value = 1\n")
    monkeypatch.setattr(
        "evaluation.swebench_workspace._archive_commit",
        lambda repository, base_commit: archive,
    )

    prepared = prepare_evaluation_workspace(tmp_path / "repository", "base", tmp_path / "result")

    assert (prepared.workspace / "package/module.py").read_text() == "value = 1\n"
    assert not (prepared.workspace / ".git").exists()
    assert prepared.session_root.is_dir()
    assert prepared.session_root.parent.parent == tmp_path / "result"
    assert prepared.session_root not in prepared.workspace.parents


def test_extract_archive_rejects_path_outside_workspace(tmp_path: Path) -> None:
    """测试归档中的越界路径不会写入评测工作区"""

    with pytest.raises(EvaluationWorkspaceError, match="越界"):
        _extract_archive(_archive_with_file("../leak.txt", "secret"), tmp_path)


def test_clean_workspace_rejects_git_directory(tmp_path: Path) -> None:
    """测试评测前会拒绝含 Git 历史的工作区"""

    (tmp_path / ".git").mkdir()

    with pytest.raises(EvaluationWorkspaceError, match=".git"):
        _assert_clean_workspace(tmp_path)
