from pathlib import Path

import core.project_instructions as project_instructions


def test_loads_parent_and_workspace_instructions_from_outer_to_inner(
    tmp_path: Path,
) -> None:
    """测试只按由外到内顺序加载直接父目录与工作区说明。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "AGENTS.md").write_text("parent", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("workspace", encoding="utf-8")
    (tmp_path.parent / "AGENTS.md").write_text("outside", encoding="utf-8")

    result = project_instructions.load_project_instructions(workspace)

    assert result.content.index("parent") < result.content.index("workspace")
    assert "outside" not in result.content
    assert result.sources == (tmp_path / "AGENTS.md", workspace / "AGENTS.md")


def test_load_project_instructions_safely_skips_missing_and_invalid_utf8(
    tmp_path: Path,
) -> None:
    """测试缺失或编码异常的项目说明会安全降级。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_bytes(b"\xff\xfe")

    result = project_instructions.load_project_instructions(workspace)

    assert result == project_instructions.ProjectInstructions("", ())


def test_load_project_instructions_marks_truncated_content(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """测试项目说明超过固定字节上限时会标记截断。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("abcdefgh", encoding="utf-8")
    monkeypatch.setattr(project_instructions, "MAX_FILE_BYTES", 4)
    monkeypatch.setattr(project_instructions, "MAX_TOTAL_BYTES", 4)

    result = project_instructions.load_project_instructions(workspace)

    assert "abcd" in result.content
    assert "[Content truncated]" in result.content
