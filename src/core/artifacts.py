"""按项目保存工具输出原文，供上下文降级后按需取回。

Artifact Store 与工具输出防火墙（Firewall）、陈旧输出驱逐（Eviction）配对：
超阈值或被驱逐的工具输出原文完整落盘，模型上下文中只保留有界占位符，
需要时通过 read_artifact 工具按行范围取回。JSONL 会话原文永不改写。
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .model import ToolCall, ToolResult
from .tools.args import optional_positive_integer, string_argument
from .tools.output_limits import limit_tool_output
from .tools.types import ToolDefinition, ToolHandler

# 超过该字符数的工具输出原文落盘，上下文只注入有界占位符
FIREWALL_THRESHOLD_CHARS = 8_000
# 占位符预览保留的头部与尾部行数
PREVIEW_HEAD_LINES = 10
PREVIEW_TAIL_LINES = 10
PREVIEW_MAX_CHARS = 2_000
ARTIFACT_MARKER = "[artifact "
DEFAULT_ARTIFACT_READ_LINES = 400


@dataclass(frozen=True)
class Artifact:
    """一份已落盘的工具输出原文及其元数据。"""

    artifact_id: str
    session_id: str
    source_tool: str
    content: str


class ArtifactStore:
    """在工作区 .epsilon/artifacts 下按项目保存输出原文。

    元数据携带 session_id，便于后续按会话审计归属；文件名使用
    内容摘要，同一原文只落盘一份。
    """

    def __init__(self, artifacts_root: Path) -> None:
        """保存 artifacts 根目录，不提前创建。"""

        self._root = artifacts_root

    @classmethod
    def for_workspace(cls, workspace: Path) -> "ArtifactStore":
        """创建生产工作区默认位置的项目级 store。"""

        return cls(workspace / ".epsilon" / "artifacts")

    def save(self, content: str, *, session_id: str, source_tool: str) -> str:
        """完整保存原文并返回 artifact id，内容不变时复用已有文件。"""

        artifact_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        path = self._path(artifact_id)
        if path.exists():
            return artifact_id
        record = {
            "id": artifact_id,
            "session_id": session_id,
            "source_tool": source_tool,
            "content": content,
        }
        self._root.mkdir(parents=True, exist_ok=True)
        # 先写临时文件再原子重命名，避免读侧看到半截 JSON
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self._root, delete=False
        ) as tmp:
            json.dump(record, tmp, ensure_ascii=False)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        return artifact_id

    def load(self, artifact_id: str) -> Artifact | None:
        """读取一份原文，id 不存在或内容损坏时返回 None。"""

        path = self._path(artifact_id)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(record, dict) or record.get("id") != artifact_id:
            return None
        content = record.get("content")
        if not isinstance(content, str):
            return None
        return Artifact(
            artifact_id=artifact_id,
            session_id=str(record.get("session_id", "")),
            source_tool=str(record.get("source_tool", "")),
            content=content,
        )

    def _path(self, artifact_id: str) -> Path:
        """生成 artifact 文件路径，id 只允许十六进制字符。"""

        if not all(char in "0123456789abcdef" for char in artifact_id):
            raise ValueError("invalid artifact id")
        return self._root / f"{artifact_id}.json"


def artifact_placeholder(artifact_id: str, original_chars: int, content: str) -> str:
    """生成注入模型上下文的有界占位符，预览保留头尾行。"""

    lines = content.splitlines()
    head = lines[:PREVIEW_HEAD_LINES]
    tail = lines[-PREVIEW_TAIL_LINES:] if len(lines) > PREVIEW_HEAD_LINES + PREVIEW_TAIL_LINES else []
    preview = "\n".join(head)
    if tail:
        preview += "\n…\n" + "\n".join(tail)
    if len(preview) > PREVIEW_MAX_CHARS:
        preview = preview[:PREVIEW_MAX_CHARS].rstrip() + "…"
    return (
        f"{preview}\n{ARTIFACT_MARKER}{artifact_id}] Full output "
        f"({original_chars} chars, {len(lines)} lines) stored. "
        "Use read_artifact with this id to retrieve any line range."
    )


def is_artifact_placeholder(content: str) -> bool:
    """判断工具输出是否已是 artifact 占位符，避免重复落盘或重复驱逐。"""

    return ARTIFACT_MARKER in content


def apply_output_firewall(
    content: str,
    store: ArtifactStore | None,
    session_id: str,
    source_tool: str,
) -> str:
    """工具输出防火墙：进入历史前定型，超阈值原文落盘并注入有界占位符。

    store 缺失时退化为原有字节/行数硬截断；占位符一旦写入历史，
    之后不再改写，保证 DeepSeek 前缀缓存只断在工具结果进入历史那一刻。
    """

    if store is not None and len(content) > FIREWALL_THRESHOLD_CHARS:
        artifact_id = store.save(
            content, session_id=session_id, source_tool=source_tool
        )
        return artifact_placeholder(artifact_id, len(content), content)
    return limit_tool_output(content)


def create_read_artifact_tool(store: ArtifactStore) -> tuple[ToolDefinition, ToolHandler]:
    """创建按行范围取回落盘原文的只读工具。"""

    async def read_artifact(tool_call: ToolCall) -> ToolResult:
        artifact_id = string_argument(tool_call, "artifact_id")
        offset = optional_positive_integer(tool_call, "offset", 1)
        limit = optional_positive_integer(tool_call, "limit", DEFAULT_ARTIFACT_READ_LINES)
        artifact = store.load(artifact_id)
        if artifact is None:
            raise ValueError(f"artifact {artifact_id} not found")
        lines = artifact.content.splitlines(keepends=True)
        if lines and offset > len(lines):
            raise ValueError(f"offset {offset} exceeds artifact line count {len(lines)}")
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
            name="read_artifact",
            description=(
                "Retrieve stored tool output by artifact id. Large or evicted tool "
                "outputs are stored as artifacts; use offset and limit to read "
                "specific line ranges."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 1, "default": 1},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "default": DEFAULT_ARTIFACT_READ_LINES,
                    },
                },
                "required": ["artifact_id"],
            },
            source="local",
            permission="read",
            idempotent=True,
            execution_mode="parallel",
        ),
        read_artifact,
    )
