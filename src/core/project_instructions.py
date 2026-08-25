"""读取工作区及直接父目录提供的项目说明。"""

from dataclasses import dataclass
from pathlib import Path


INSTRUCTION_FILENAME = "AGENTS.md"
MAX_FILE_BYTES = 32 * 1024
MAX_TOTAL_BYTES = 64 * 1024


@dataclass(frozen=True)
class ProjectInstructions:
    """保存可注入模型上下文的项目说明和来源。"""

    content: str
    sources: tuple[Path, ...]


def load_project_instructions(workspace: Path) -> ProjectInstructions:
    """按由外到内顺序读取工作区及直接父目录的 AGENTS.md。"""

    root = workspace.resolve()
    remaining = MAX_TOTAL_BYTES
    sections: list[str] = []
    sources: list[Path] = []
    for path in (root.parent / INSTRUCTION_FILENAME, root / INSTRUCTION_FILENAME):
        if remaining <= 0:
            break
        content, truncated = _read_instruction(path, min(MAX_FILE_BYTES, remaining))
        if content is None:
            continue
        suffix = "\n\n[Content truncated]" if truncated else ""
        sections.append(
            f"### Source: {_source_label(path, root)}\n\n{content}{suffix}"
        )
        sources.append(path)
        remaining -= len(content.encode("utf-8"))
    return ProjectInstructions("\n\n".join(sections), tuple(sources))


def _read_instruction(path: Path, limit: int) -> tuple[str | None, bool]:
    """在固定字节上限内读取 UTF-8 项目说明，失败时静默跳过。"""

    try:
        with path.open("rb") as file:
            data = file.read(limit + 1)
    except OSError:
        return None, False
    truncated = len(data) > limit
    try:
        content = data[:limit].decode("utf-8").strip()
    except UnicodeDecodeError:
        return None, False
    return (content or None), truncated


def _source_label(path: Path, workspace: Path) -> str:
    """返回不泄露工作区外绝对路径的来源标识。"""

    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return f"../{path.name}"
