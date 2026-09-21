"""按会话保存工具输出原文，供上下文降级后按需取回。

Artifact Store 与工具输出防火墙（Firewall）、陈旧输出驱逐（Eviction）配对：
超阈值或被驱逐的工具输出原文以纯文本落盘，模型上下文中只保留有界占位符，
需要时通过 read_file 读取 artifact://<id> 按行范围取回。JSONL 会话原文永不改写。
"""

from dataclasses import dataclass
from pathlib import Path

# 超过该字符数的工具输出原文落盘，上下文只注入有界占位符
FIREWALL_THRESHOLD_CHARS = 8_000
# 占位符预览保留的头部与尾部行数
PREVIEW_HEAD_LINES = 10
PREVIEW_TAIL_LINES = 10
PREVIEW_MAX_CHARS = 2_000
# 模型读取工具输出原文的引用前缀
ARTIFACT_URL_PREFIX = "artifact://"


@dataclass(frozen=True)
class Artifact:
    """一份已落盘的工具输出原文及其元数据。"""

    artifact_id: str
    session_id: str
    source_tool: str
    content: str


class ArtifactStore:
    """在工作区 .epsilon/artifacts 下按会话保存输出原文。

    文件布局 artifacts/<session_id>/<id>.<tool_type>.txt，id 为会话内递增序号，
    会话归属由子目录表达。set_session_id 绑定后，load/resolve 默认在该会话目录内查找。
    """

    def __init__(self, artifacts_root: Path) -> None:
        """保存 artifacts 根目录与当前会话绑定。"""

        self._root = artifacts_root
        self._session_id = ""

    @classmethod
    def for_workspace(cls, workspace: Path) -> "ArtifactStore":
        """创建生产工作区默认位置的项目级 store。"""

        return cls(workspace / ".epsilon" / "artifacts")

    def set_session_id(self, session_id: str) -> None:
        """绑定当前会话，load/resolve 默认在其会话目录内查找。"""

        self._session_id = session_id

    def save(self, content: str, *, session_id: str, source_tool: str) -> str:
        """以纯文本落盘原文并返回会话内递增序号 id。"""

        directory = self._root / session_id
        directory.mkdir(parents=True, exist_ok=True)
        artifact_id = _next_artifact_id(directory)
        path = directory / f"{artifact_id}.{_safe_tool_type(source_tool)}.txt"
        path.write_text(content, encoding="utf-8")
        return str(artifact_id)

    def resolve(self, artifact_id: str, session_id: str | None = None) -> Path | None:
        """把 artifact id 解析为落盘路径，未知或非法 id 返回 None。"""

        target = session_id if session_id is not None else self._session_id
        if not target or not str(artifact_id).isdigit():
            return None
        matches = sorted((self._root / target).glob(f"{artifact_id}.*.txt"))
        return matches[0] if matches else None

    def load(self, artifact_id: str, session_id: str | None = None) -> Artifact | None:
        """读取一份原文，id 不存在或文件不可读时返回 None。"""

        path = self.resolve(artifact_id, session_id)
        if path is None:
            return None
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None
        target = session_id if session_id is not None else self._session_id
        return Artifact(
            artifact_id=str(artifact_id),
            session_id=target,
            source_tool=_tool_type_from_name(path.name),
            content=content,
        )


def _next_artifact_id(directory: Path) -> int:
    """扫描会话目录得到下一个递增序号。"""

    highest = 0
    for path in directory.glob("*.txt"):
        stem = path.name.split(".", 1)[0]
        if stem.isdigit():
            highest = max(highest, int(stem))
    return highest + 1


def _safe_tool_type(source_tool: str) -> str:
    """把工具名转换为只含文件安全字符的类型片段。"""

    return "".join(
        char if char.isalnum() or char in "_-" else "_" for char in source_tool
    ) or "tool"


def _tool_type_from_name(filename: str) -> str:
    """从 <id>.<tool_type>.txt 文件名还原工具类型。"""

    parts = filename.split(".", 2)
    return parts[1] if len(parts) > 1 else ""


def artifact_placeholder(artifact_id: str, content: str) -> str:
    """生成注入模型上下文的有界占位符，预览保留头尾行并给出取回地址。"""

    lines = content.splitlines()
    head = lines[:PREVIEW_HEAD_LINES]
    tail = (
        lines[-PREVIEW_TAIL_LINES:]
        if len(lines) > PREVIEW_HEAD_LINES + PREVIEW_TAIL_LINES
        else []
    )
    preview = "\n".join(head)
    if tail:
        preview += "\n…\n" + "\n".join(tail)
    if len(preview) > PREVIEW_MAX_CHARS:
        preview = preview[:PREVIEW_MAX_CHARS].rstrip() + "…"
    return f"{preview}\nFull output: {ARTIFACT_URL_PREFIX}{artifact_id}"


def is_artifact_placeholder(content: str) -> bool:
    """判断工具输出是否已是 artifact 占位符，避免重复落盘或重复驱逐。"""

    return ARTIFACT_URL_PREFIX in content


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
        return artifact_placeholder(artifact_id, content)
    # 延迟导入，避免 artifacts 与 tools 包在模块加载期相互引用
    from .tools.output_limits import limit_tool_output

    return limit_tool_output(content)
