from pathlib import Path

import pytest

from core.model import ToolCall
from core.tools import (
    MAX_TOOL_OUTPUT_BYTES,
    MAX_TOOL_OUTPUT_LINES,
    TRUNCATION_NOTICE,
    ToolManager,
    WorkspacePathError,
    create_list_files_tool,
    create_read_file_tool,
    create_search_files_tool,
)
from core.tools.file_tools import MAX_FILE_READ_BYTES


def _call(name: str, arguments: dict[str, object]) -> ToolCall:
    """构造测试用工具调用。"""

    return ToolCall(call_id="call-1", name=name, arguments=arguments)


def _manager(
    workspace: Path,
    *tools: tuple,
) -> ToolManager:
    """注册指定的本地文件工具。"""

    manager = ToolManager()
    for definition, handler in tools:
        manager.register_local(definition, handler)
    return manager


@pytest.mark.asyncio
async def test_read_file_returns_complete_text(tmp_path: Path) -> None:
    """测试读取工具返回完整文件内容。"""

    (tmp_path / "README.md").write_text("第一行\n第二行", encoding="utf-8")
    manager = _manager(tmp_path, create_read_file_tool(tmp_path))

    result = await manager.execute(_call("read_file", {"path": "README.md"}))

    assert result.content == "第一行\n第二行"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_read_file_returns_requested_line_range_and_continuation(tmp_path: Path) -> None:
    """测试读取工具按行返回范围，并提示下一次读取位置"""

    (tmp_path / "README.md").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    manager = _manager(tmp_path, create_read_file_tool(tmp_path))

    result = await manager.execute(
        _call("read_file", {"path": "README.md", "offset": 2, "limit": 2})
    )

    assert result.content == (
        "two\nthree\n\n"
        "[Showing lines 2-3 of 4. Use offset=4 to continue.]"
    )
    assert result.is_error is False


@pytest.mark.asyncio
async def test_read_file_rejects_unknown_or_invalid_range_arguments(tmp_path: Path) -> None:
    """测试本地工具不会静默忽略未知参数和非法分页参数"""

    (tmp_path / "README.md").write_text("one\n", encoding="utf-8")
    manager = _manager(tmp_path, create_read_file_tool(tmp_path))

    unknown = await manager.execute(
        _call("read_file", {"path": "README.md", "unused": True})
    )
    invalid = await manager.execute(
        _call("read_file", {"path": "README.md", "offset": 0})
    )

    assert unknown.is_error is True
    assert unknown.error_category == "invalid_request"
    assert "unknown tool argument: unused" in unknown.content
    assert invalid.is_error is True
    assert invalid.error_category == "invalid_request"
    assert "argument offset must be >= 1" in invalid.content


@pytest.mark.asyncio
async def test_read_file_truncates_large_output(tmp_path: Path) -> None:
    """测试读取工具会限制返回给模型的文本大小。"""

    (tmp_path / "large.txt").write_text("x" * (MAX_TOOL_OUTPUT_BYTES + 1))
    manager = _manager(tmp_path, create_read_file_tool(tmp_path))

    result = await manager.execute(_call("read_file", {"path": "large.txt"}))

    assert result.content.endswith(TRUNCATION_NOTICE)
    assert len(result.content.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES


@pytest.mark.asyncio
async def test_read_file_rejects_file_larger_than_read_limit(tmp_path: Path) -> None:
    """测试读取工具会在读入前拒绝超大文件。"""

    (tmp_path / "too-large.txt").write_bytes(b"x" * (MAX_FILE_READ_BYTES + 1))
    manager = _manager(tmp_path, create_read_file_tool(tmp_path))

    result = await manager.execute(_call("read_file", {"path": "too-large.txt"}))

    assert result.is_error is True
    assert "1 MB read limit" in result.content


@pytest.mark.asyncio
async def test_list_files_returns_sorted_entries(tmp_path: Path) -> None:
    """测试目录列表按名称排序并标记子目录。"""

    (tmp_path / "b.txt").write_text("", encoding="utf-8")
    (tmp_path / "a").mkdir()
    manager = _manager(tmp_path, create_list_files_tool(tmp_path))

    result = await manager.execute(_call("list_files", {}))

    assert result.content == "a/\nb.txt"


@pytest.mark.asyncio
async def test_search_files_returns_matching_lines(tmp_path: Path) -> None:
    """测试搜索工具返回文件路径、行号和匹配行。"""

    (tmp_path / "main.py").write_text("one\nneedle here\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("nothing\n", encoding="utf-8")
    manager = _manager(tmp_path, create_search_files_tool(tmp_path))

    result = await manager.execute(
        _call("search_files", {"pattern": "needle"})
    )

    assert result.content == "main.py:2: needle here"


@pytest.mark.asyncio
async def test_search_files_truncates_excessive_matches(tmp_path: Path) -> None:
    """测试搜索工具会限制过多匹配结果。"""

    (tmp_path / "matches.txt").write_text("needle\n" * (MAX_TOOL_OUTPUT_LINES + 1))
    manager = _manager(tmp_path, create_search_files_tool(tmp_path))

    result = await manager.execute(_call("search_files", {"pattern": "needle"}))

    assert result.content.endswith(TRUNCATION_NOTICE)


@pytest.mark.asyncio
async def test_search_files_skips_ignored_binary_and_oversized_files(tmp_path: Path) -> None:
    """测试搜索跳过常见运行目录、二进制和超大文件。"""

    (tmp_path / "main.py").write_text("needle\n", encoding="utf-8")
    ignored = tmp_path / ".git"
    ignored.mkdir()
    (ignored / "config").write_text("needle\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"needle\0")
    (tmp_path / "large.txt").write_bytes(
        b"needle" + b"x" * MAX_FILE_READ_BYTES
    )
    manager = _manager(tmp_path, create_search_files_tool(tmp_path))

    result = await manager.execute(_call("search_files", {"pattern": "needle"}))

    assert result.content == "main.py:1: needle"


@pytest.mark.asyncio
async def test_file_tools_reject_workspace_escape(tmp_path: Path) -> None:
    """测试文件工具拒绝访问工作区之外的路径。"""

    manager = _manager(tmp_path, create_read_file_tool(tmp_path))

    result = await manager.execute(_call("read_file", {"path": "../secret.txt"}))

    assert result.is_error is True
    assert "inside the current workspace" in result.content


def test_workspace_path_resolver_rejects_absolute_escape(tmp_path: Path) -> None:
    """测试工作区路径解析器拒绝工作区之外的绝对路径。"""

    from core.tools import resolve_workspace_path

    with pytest.raises(WorkspacePathError):
        resolve_workspace_path(tmp_path, "/tmp/secret.txt")
