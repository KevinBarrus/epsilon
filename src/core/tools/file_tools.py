"""实现第一批只读本地文件工具。"""

import os
from pathlib import Path

from ..model import ToolCall, ToolResult
from .args import optional_path, optional_positive_integer, string_argument
from .output_limits import limit_tool_output
from .path_utils import resolve_workspace_path
from .types import ToolDefinition, ToolHandler


MAX_FILE_READ_BYTES = 1_000_000
DEFAULT_FILE_READ_LINES = 400
IGNORED_SEARCH_DIRECTORIES = {".git", ".epsilon", ".venv", "node_modules"}


def create_read_file_tool(workspace: Path) -> tuple[ToolDefinition, ToolHandler]:
    """创建读取单个文件的工具。"""

    async def read_file(tool_call: ToolCall) -> ToolResult:
        path = resolve_workspace_path(workspace, string_argument(tool_call, "path"))
        offset = optional_positive_integer(tool_call, "offset", 1)
        limit = optional_positive_integer(tool_call, "limit", DEFAULT_FILE_READ_LINES)
        if not path.is_file():
            raise ValueError("target is not a file")
        if path.stat().st_size > MAX_FILE_READ_BYTES:
            raise ValueError("file exceeds the 1 MB read limit")
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if lines and offset > len(lines):
            raise ValueError(f"offset {offset} exceeds file line count {len(lines)}")
        selected = lines[offset - 1 : offset - 1 + limit]
        content = "".join(selected)
        last_line = offset + len(selected) - 1
        if last_line < len(lines):
            content = (
                f"{content.rstrip()}\n\n"
                f"[Showing lines {offset}-{last_line} of {len(lines)}. "
                f"Use offset={last_line + 1} to continue.]"
            )
        return ToolResult(
            call_id=tool_call.call_id,
            content=limit_tool_output(content),
        )

    return (
        ToolDefinition(
            name="read_file",
            description=(
                "Read text lines from a file in the workspace. Use offset and limit "
                "to read large files in parts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 1, "default": 1},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "default": DEFAULT_FILE_READ_LINES,
                    },
                },
                "required": ["path"],
            },
            source="local",
            permission="read",
            idempotent=True,
            execution_mode="parallel",
            capability="file.read",
        ),
        read_file,
    )


def create_list_files_tool(workspace: Path) -> tuple[ToolDefinition, ToolHandler]:
    """创建列出目录内容的工具。"""

    async def list_files(tool_call: ToolCall) -> ToolResult:
        path = resolve_workspace_path(workspace, optional_path(tool_call))
        if not path.is_dir():
            raise ValueError("target is not a directory")

        entries = sorted(path.iterdir(), key=lambda item: item.name)
        content = "\n".join(
            f"{entry.name}{'/' if entry.is_dir() else ''}" for entry in entries
        )
        return ToolResult(
            call_id=tool_call.call_id,
            content=limit_tool_output(content) if content else "directory is empty",
        )

    return (
        ToolDefinition(
            name="list_files",
            description="List the direct contents of a directory in the workspace",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                },
            },
            source="local",
            permission="read",
            idempotent=True,
            execution_mode="parallel",
            capability="file.read",
        ),
        list_files,
    )


def create_search_files_tool(workspace: Path) -> tuple[ToolDefinition, ToolHandler]:
    """创建在工作区内按文本查找内容的工具。"""

    async def search_files(tool_call: ToolCall) -> ToolResult:
        pattern = string_argument(tool_call, "pattern")
        root = resolve_workspace_path(workspace, optional_path(tool_call))
        if not root.is_dir():
            raise ValueError("search scope is not a directory")

        matches: list[str] = []
        for directory, directories, filenames in os.walk(root):
            directories[:] = sorted(
                name for name in directories if name not in IGNORED_SEARCH_DIRECTORIES
            )
            for filename in sorted(filenames):
                path = Path(directory, filename)
                if not _is_searchable_file(path):
                    continue
                try:
                    with path.open(encoding="utf-8", errors="replace") as file:
                        for line_number, line in enumerate(file, start=1):
                            if pattern not in line:
                                continue
                            relative_path = path.relative_to(workspace.resolve())
                            match = f"{relative_path}:{line_number}: {line.rstrip()}"
                            content = "\n".join([*matches, match])
                            limited = limit_tool_output(content)
                            if limited != content:
                                return ToolResult(call_id=tool_call.call_id, content=limited)
                            matches.append(match)
                except OSError:
                    continue

        return ToolResult(
            call_id=tool_call.call_id,
            content=limit_tool_output("\n".join(matches)) if matches else "no matching content found",
        )

    return (
        ToolDefinition(
            name="search_files",
            description="Search workspace files by text content",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                },
                "required": ["pattern"],
            },
            source="local",
            permission="read",
            idempotent=True,
            execution_mode="parallel",
            capability="file.read",
        ),
        search_files,
    )


def _is_searchable_file(path: Path) -> bool:
    """判断文件是否适合按文本逐行搜索。"""

    try:
        if path.stat().st_size > MAX_FILE_READ_BYTES:
            return False
        with path.open("rb") as file:
            return b"\0" not in file.read(4_096)
    except OSError:
        return False
