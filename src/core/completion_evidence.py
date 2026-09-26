"""完成门的证据包与脚本化检查器。

**核心原则：验证是「判定一份有限的、宿主预先装配好的证据」，不是「再探索一遍」。**

v1 的错在于把验证器做成一个只读子 Agent 去读整个仓库——单个大文件就能把预算烧光。
v2 由宿主装配一份**有硬上限**的证据简报（整包 ≤ 14k 字符），验证器默认**单次调用、不给工具**，
只能依据这份简报判定。上限口径对齐 minimax-code 的 `evidence-brief.ts`。
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .model import Message, ToolCall, ToolResult

# 证据包各项硬上限（照抄 minimax-code 口径）
MAX_BRIEF_CHARS = 14_000
MAX_OBJECTIVE_CHARS = 4_000
MAX_CRITERIA_CHARS = 4_000
MAX_CLAIM_CHARS = 2_000
MAX_CHANGE_CHARS = 4_000
MAX_CHANGE_ITEMS = 100
MAX_COMMAND_CHARS = 4_000
MAX_COMMAND_ITEMS = 100
MAX_CHECK_CHARS = 4_000
MAX_TAIL_CHARS = 4_000
MAX_TAIL_MESSAGES = 5
# 读交付物时也不无限读
MAX_DELIVERABLE_CHARS = 200_000
CHECK_TIMEOUT_SECONDS = 60

# 能体现"改动"与"命令"的工具
WRITE_TOOLS = frozenset({"write_file", "edit_file", "apply_patch"})
COMMAND_TOOLS = frozenset({"run_command"})
_PATH_TOOLS = frozenset({"read_file", "write_file", "edit_file"})

_EXIT_CODE = re.compile(r"exit code:\s*(-?\d+)")
_FILE_PATH = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]{1,8}")


def _clip(text: str, limit: int) -> str:
    """截断到上限，超出时给出明确标记。"""

    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 12)] + "…[truncated]"


@dataclass(frozen=True)
class CheckResult:
    """一条脚本化检查的结果。

    三态：`passed` / `failed` / `unverified`。第三态用于"核验没能做完"（例如核验次数超上限），
    它**既不算通过也不算失败**——必须让上层按"未取得证据"处理，不能静默通过。
    """

    kind: str
    passed: bool
    detail: str = ""
    unverified: bool = False

    @property
    def status(self) -> str:
        """三态之一：passed / failed / unverified。"""

        if self.unverified:
            return "unverified"
        return "passed" if self.passed else "failed"


@dataclass(frozen=True)
class EvidenceBundle:
    """宿主装配好的有界证据。"""

    objective: str = ""
    criteria: str = ""
    claim: str = ""
    changes: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    checks: tuple[CheckResult, ...] = ()
    tail: tuple[str, ...] = ()
    unscripted: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        """按优先级拼出 ≤ MAX_BRIEF_CHARS 的证据简报。

        优先级（高→低）：objective → criteria → checks → claim → changes → commands → tail；
        超限时从低优先级开始裁。
        """

        sections: list[tuple[str, str]] = [
            ("目标", _clip(self.objective, MAX_OBJECTIVE_CHARS)),
            ("验收标准（完成前必须满足）", _clip(self.criteria, MAX_CRITERIA_CHARS)),
        ]
        if self.checks:
            lines = [f"[{r.status.upper()}] {r.kind}: {r.detail}" for r in self.checks]
            sections.append(("脚本化检查结果（宿主执行）", _clip("\n".join(lines), MAX_CHECK_CHARS)))
        if self.unscripted:
            sections.append(
                (
                    "无脚本证据的验收条目（对这类条目一律判 inconclusive）",
                    _clip("\n".join(f"- {item}" for item in self.unscripted), MAX_CRITERIA_CHARS),
                )
            )
        if self.claim:
            sections.append(("完成声称", _clip(self.claim, MAX_CLAIM_CHARS)))
        if self.changes:
            sections.append(
                ("改动的文件（来自动作记录）", _clip("\n".join(self.changes), MAX_CHANGE_CHARS))
            )
        if self.commands:
            sections.append(
                ("跑过的命令与退出码（来自动作记录）", _clip("\n".join(self.commands), MAX_COMMAND_CHARS))
            )
        if self.tail:
            sections.append(("近期对话尾巴", _clip("\n".join(self.tail), MAX_TAIL_CHARS)))

        low_to_high = ["近期对话尾巴", "跑过的命令与退出码（来自动作记录）", "改动的文件（来自动作记录）"]
        while len(_join(sections)) > MAX_BRIEF_CHARS:
            for name in low_to_high:
                index = next((i for i, (title, _) in enumerate(sections) if title == name), None)
                if index is not None:
                    sections.pop(index)
                    break
            else:
                break  # 没有可裁的低优先级小节了
        return _clip(_join(sections), MAX_BRIEF_CHARS)


def _join(sections: Sequence[tuple[str, str]]) -> str:
    """把小节拼成简报文本。"""

    blocks = [f"## {title}\n{body}" for title, body in sections if body]
    return "\n\n".join(blocks)


def extract_changes(
    tool_calls: Sequence[ToolCall],
    results: Sequence[ToolResult],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """从工具调用与结果里提取"改了哪些文件、跑了哪些命令"（含成功/失败标记）。

    证据来自 **Agent 的动作记录**，不是重新读代码——这是 v2 与 v1 的根本差别。
    """

    changes: list[str] = []
    commands: list[str] = []
    for call, result in zip(tool_calls, results):
        status = "failed" if result.is_error else "ok"
        if call.name in WRITE_TOOLS:
            path = call.arguments.get("path") or call.arguments.get("file_path") or "?"
            changes.append(f"{status} {call.name} {path}")
        elif call.name in COMMAND_TOOLS:
            command = str(call.arguments.get("command", "?"))
            match = _EXIT_CODE.search(result.content or "")
            code = match.group(1) if match else "0"
            commands.append(f"{status} exit={code} {_clip(command, 160)}")
    return (
        tuple(changes[:MAX_CHANGE_ITEMS]),
        tuple(commands[:MAX_COMMAND_ITEMS]),
    )


def recent_tail(messages: Sequence[Message], limit: int = MAX_TAIL_MESSAGES) -> tuple[str, ...]:
    """取最后若干条消息的简短文本，作为"近期对话尾巴"。"""

    tail: list[str] = []
    for message in list(messages)[-limit:]:
        text = _clip(f"{message.role}: {message.content}", 400)
        if text:
            tail.append(text)
    return tuple(tail)


def _read_deliverable(path: Path) -> str:
    """有界读取交付物（只用于脚本化检查，不读代码库）。"""

    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:MAX_DELIVERABLE_CHARS]
    except OSError:
        return ""


def _existing_paths(workspace: Path) -> set[str]:
    """收集工作区里的相对路径，用于 path_exists 判定（有界遍历）。"""

    found: set[str] = set()
    for path in workspace.rglob("*"):
        if path.is_file():
            found.add(path.relative_to(workspace).as_posix())
    return found


def _mentions(text: str) -> list[str]:
    """从文本里抽出可能的文件路径 token。"""

    seen: dict[str, None] = {}
    for match in _FILE_PATH.finditer(text):
        seen.setdefault(match.group(0), None)
    return list(seen)


def run_checks(
    specs: Sequence[Mapping[str, object]],
    workspace: Path,
    changes: Sequence[str] = (),
    existing: set[str] | None = None,
) -> tuple[CheckResult, ...]:
    """执行脚本化检查，返回 pass/fail 与证据。

    内置检查器都是有界的：只看交付物与工作区相对路径，不读代码库全文。
    """

    paths = _existing_paths(workspace) if existing is None else existing
    results: list[CheckResult] = []
    for spec in specs:
        kind = str(spec.get("kind", ""))
        if kind == "path_exists":
            targets = _as_str_list(spec.get("paths"))
            missing = [p for p in targets if p not in paths]
            results.append(
                CheckResult(kind, not missing, "全部存在" if not missing else f"缺失：{missing}")
            )
        elif kind == "sections_cover":
            deliverable = _read_deliverable(workspace / str(spec.get("path", "")))
            sections = _as_str_list(spec.get("sections"))
            missing = [s for s in sections if s not in deliverable]
            results.append(
                CheckResult(
                    kind,
                    bool(sections) and not missing,
                    f"{len(sections) - len(missing)}/{len(sections)} 个小节存在"
                    + (f"，缺：{missing}" if missing else ""),
                )
            )
        elif kind == "paths_per_section":
            deliverable = _read_deliverable(workspace / str(spec.get("path", "")))
            sections = _as_str_list(spec.get("sections"))
            minimum = int(spec.get("min_paths", 1))
            weak: list[str] = []
            for section in sections:
                body = _section_body(deliverable, section)
                real = [p for p in _mentions(body) if p in paths]
                if len(real) < minimum:
                    weak.append(f"{section}({len(real)}/{minimum})")
            results.append(
                CheckResult(
                    kind,
                    bool(sections) and not weak,
                    "每节都达到要求" if not weak else f"不足：{weak}",
                )
            )
        elif kind == "mentioned_paths_exist":
            # 通用检查：交付物里提到的路径必须真实存在（不泄露任何隐藏清单）
            deliverable = _read_deliverable(workspace / str(spec.get("path", "")))
            mentioned = _mentions(deliverable)
            missing = [p for p in mentioned if p not in paths]
            results.append(
                CheckResult(
                    kind,
                    bool(mentioned) and not missing,
                    f"{len(mentioned) - len(missing)}/{len(mentioned)} 个路径存在"
                    + (f"，编造：{missing[:10]}" if missing else ""),
                )
            )
        elif kind == "files_changed":
            targets = _as_str_list(spec.get("paths"))
            joined = "\n".join(changes)
            missing = [p for p in targets if p not in joined]
            results.append(
                CheckResult(kind, not missing, "都已改动" if not missing else f"未改动：{missing}")
            )
        elif kind == "command_passed":
            command = str(spec.get("command", "")).strip()
            if not command:
                results.append(CheckResult(kind, False, "未提供命令"))
                continue
            try:
                completed = subprocess.run(
                    command,
                    shell=True,
                    cwd=workspace,
                    capture_output=True,
                    timeout=CHECK_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                results.append(CheckResult(kind, False, f"超时 {CHECK_TIMEOUT_SECONDS}s：{command}"))
            else:
                results.append(
                    CheckResult(
                        kind,
                        completed.returncode == 0,
                        f"exit={completed.returncode} {_clip(command, 120)}",
                    )
                )
        elif kind == "identifiers_per_section":
            # v3 命门：每节必须举出**在代码里真实存在**的标识符（常量/类型/函数名）。
            # 要凑齐这三类且都被核验存在，只能真去读那部分代码。
            name = str(spec.get("path", ""))
            deliverable = _read_deliverable(workspace / name)
            sections = _as_str_list(spec.get("sections"))
            minimum = int(spec.get("min_identifiers", 3))
            categories = _as_str_list(spec.get("categories")) or list(IDENTIFIER_CATEGORIES)
            cap = int(spec.get("max_lookups", MAX_IDENTIFIER_LOOKUPS))
            # 交付物与本次被写过的文件排除出语料：否则自己写的名字就「自动存在」了
            index = CorpusIndex(workspace, exclude=[name, *_changed_paths(changes)])
            seen: set[str] = set()  # 同一标识符跨节不重复计数
            lookups = 0
            overflow = False
            weak: list[str] = []
            for section in sections:
                body = _section_body(deliverable, section)
                counts = {category: 0 for category in categories}
                verified = 0
                for token in _identifiers(body):
                    if token in seen:
                        continue
                    if lookups >= cap:
                        overflow = True
                        break
                    lookups += 1
                    category = classify_identifier(token)
                    if category is None or not index.contains(token):
                        continue
                    seen.add(token)
                    verified += 1
                    if category in counts:
                        counts[category] += 1
                missing = [c for c in categories if counts[c] < 1]
                if verified < minimum or missing:
                    weak.append(f"{section}({verified}/{minimum}，缺类别 {missing})")
                if overflow:
                    break
            if overflow:
                results.append(
                    CheckResult(
                        kind,
                        False,
                        f"核验次数超过上限 {cap}，未能完成存在性核验",
                        unverified=True,
                    )
                )
            else:
                results.append(
                    CheckResult(
                        kind,
                        bool(sections) and not weak,
                        "每节都举出了真实存在的标识符"
                        if not weak
                        else f"不足：{weak}",
                    )
                )
        elif kind == "question_types_per_section":
            # 每节必须真的回答了那 5 类问题：命中关键词**并且**该段引用了代码级证据
            name = str(spec.get("path", ""))
            deliverable = _read_deliverable(workspace / name)
            sections = _as_str_list(spec.get("sections"))
            types = _as_str_list(spec.get("question_types")) or list(QUESTION_TYPES)
            weak: list[str] = []
            for section in sections:
                segments = _question_segments(_section_body(deliverable, section))
                missing = []
                for name_of_type in types:
                    keywords = QUESTION_TYPES.get(name_of_type, (name_of_type,))
                    hits = [s for s in segments if any(k in s for k in keywords)]
                    if not hits or not any(has_citation(s) for s in hits):
                        missing.append(name_of_type)
                if missing:
                    weak.append(f"{section}(缺：{missing})")
            results.append(
                CheckResult(
                    kind,
                    bool(sections) and not weak,
                    "5 类问题都有代码级证据" if not weak else f"不足：{weak}",
                )
            )
        else:
            # 不认识的检查器：不假装检查过（返回失败，交由上层按 inconclusive 处理）
            results.append(CheckResult(kind or "unknown", False, "不支持的检查器"))
    return tuple(results)


def _as_str_list(value: object) -> list[str]:
    """把配置里的列表规整成字符串列表。"""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _section_body(text: str, heading: str) -> str:
    """取某个小节到下一个标题之间的正文。

    **优先匹配标题行**（以 # 开头且包含该名字）：否则会撞上正文里第一次提到的同名
    单词，把前言当成小节正文（真机踩过：明明写了 5 个真实路径却判成 0/2）。
    """

    start = -1
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and heading in stripped:
            start = text.index(line) + len(line)
            break
    if start < 0:
        start = text.find(heading)
        if start < 0:
            return ""
        start += len(heading)
    rest = text[start:]
    lines: list[str] = []
    for line in rest.splitlines():
        if line.startswith("#") and line.strip() != heading:
            break
        lines.append(line)
    return "\n".join(lines)


# 可选的 subagent 验证模式硬上限：最多读 3 个文件、每个最多 200 行
MAX_VERIFIER_FILES = 3
MAX_VERIFIER_LINES = 200


class BoundedReadGuard:
    """给"可选 subagent 验证模式"用的有界读守卫。

    默认的 evaluator 模式根本不给工具；只有显式选择 subagent 模式时才用这个守卫，
    把"验证器读整个仓库"这条路彻底堵死。
    """

    def __init__(self, max_files: int = MAX_VERIFIER_FILES, max_lines: int = MAX_VERIFIER_LINES) -> None:
        """记录上限并初始化已读文件计数。"""

        self._max_files = max_files
        self._max_lines = max_lines
        self._read: set[str] = set()

    @property
    def read_files(self) -> int:
        """已读取过的不同文件数。"""

        return len(self._read)

    def allow(self, path: str) -> str | None:
        """判断一次读取是否允许；不允许时返回拒绝原因。"""

        if path not in self._read and len(self._read) >= self._max_files:
            return f"验证器最多只能读 {self._max_files} 个文件"
        self._read.add(path)
        return None

    def clip(self, content: str) -> tuple[str, bool]:
        """把文件内容裁到行数上限内；返回 (内容, 是否被裁剪)。"""

        lines = content.splitlines()
        if len(lines) <= self._max_lines:
            return content, False
        return "\n".join(lines[: self._max_lines]), True


# ===================== v3：子系统问题模板与标识符存在性核验 =====================

# 每个子系统必须回答的 5 类问题（**模板**，不含任何具体事实，所以不泄露隐藏清单）
QUESTION_TYPES: dict[str, tuple[str, ...]] = {
    "职责": ("职责", "负责", "作用", "目的是"),
    "交互": ("交互", "依赖", "调用方", "被谁调用", "协作"),
    "关键数据结构": ("数据结构", "关键类型", "核心类型", "结构体", "字段"),
    "关键参数": ("参数", "阈值", "常量", "默认值", "上限"),
    "失败模式": ("失败模式", "错误处理", "异常", "降级", "报错"),
}

# 标识符三类：常量 / 类型 / 函数。要求每节都出现，防止"只抄一堆常量"刷分。
IDENTIFIER_CATEGORIES = ("constant", "type", "function")

# 一次核验里允许的"搜索次数"上限；超限一律按未验证处理（不得静默通过）
MAX_IDENTIFIER_LOOKUPS = 200
# 被核验的语料索引也有界
MAX_CORPUS_FILES = 20_000
MAX_CORPUS_BYTES = 64_000_000
# 代码文件扩展名（只把代码当"存在性"依据）
_CODE_SUFFIXES = frozenset(
    {
        ".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".kt", ".c", ".h",
        ".cc", ".cpp", ".hpp", ".rb", ".php", ".cs", ".swift", ".scala", ".sh", ".sql",
    }
)

_BACKTICKED = re.compile(r"`([^`\n]+)`")
_CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CONSTANT = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$|^[A-Z][A-Z0-9_]*[0-9][A-Z0-9_]*$")
_TYPE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_FUNCTION = re.compile(r"^[a-z_][a-z0-9_]{2,}$")
_NOT_IDENTIFIER = frozenset({"self", "crate", "pub", "fn", "let", "mut", "use", "impl", "true", "false", "None", "Some", "Ok", "Err"})


def _identifiers(text: str) -> list[str]:
    """从正文里抽出被反引号或代码块包裹的候选标识符（保持出现顺序、去重）。"""

    chunks: list[str] = _BACKTICKED.findall(text) + _CODE_BLOCK.findall(text)
    found: dict[str, None] = {}
    for chunk in chunks:
        for match in _IDENTIFIER.finditer(chunk):
            token = match.group(0)
            if token in _NOT_IDENTIFIER:
                continue
            found.setdefault(token, None)
    return list(found)


def classify_identifier(token: str) -> str | None:
    """把标识符归到 constant / type / function；不像标识符的返回 None。"""

    if _CONSTANT.match(token):
        return "constant"
    if _TYPE.match(token):
        return "type"
    if _FUNCTION.match(token):
        return "function"
    return None


class CorpusIndex:
    """语料里"真实出现过的标识符"索引（有界构建）。

    **只索引代码文件**，并且**排除交付物与本次被写过的文件**——
    否则模型自己写进报告的名字就"自动存在"了，存在性核验就失去意义。
    """

    def __init__(self, workspace: Path, exclude: Sequence[str] = ()) -> None:
        """构建索引；预算耗尽即停止（宁可少索引，不可无界读盘）。"""

        self._excluded = {str(item) for item in exclude}
        self.tokens: set[str] = set()
        self.files = 0
        total_bytes = 0
        self.truncated = False
        for path in sorted(workspace.rglob("*")):
            if not path.is_file() or path.suffix not in _CODE_SUFFIXES:
                continue
            relative = path.relative_to(workspace).as_posix()
            if relative in self._excluded:
                continue
            if self.files >= MAX_CORPUS_FILES or total_bytes >= MAX_CORPUS_BYTES:
                self.truncated = True
                break
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            self.files += 1
            total_bytes += len(content)
            self.tokens.update(_IDENTIFIER.findall(content))

    def contains(self, token: str) -> bool:
        """标识符是否在语料里出现过。"""

        return token in self.tokens


def _question_segments(body: str) -> list[str]:
    """把一小节正文按空行切段，便于按段判定"某类问题是否被回答"。"""

    return [chunk.strip() for chunk in re.split(r"\n\s*\n", body) if chunk.strip()]


def has_citation(segment: str) -> bool:
    """某一段是否引用了代码级证据（反引号里的标识符或路径）。"""

    return bool(_identifiers(segment)) or bool(_mentions(segment))


def _changed_paths(changes: Sequence[str]) -> list[str]:
    """从改动记录里解析出被写过的文件路径（用于把交付物排除出语料索引）。"""

    paths: list[str] = []
    for entry in changes:
        parts = str(entry).split()
        if len(parts) >= 3:
            paths.append(parts[-1])
    return paths


# 检查器规格：未知检查器或参数缺失/类型不对 → 设置时报错（不许静默跳过）
CHECK_SPECS: dict[str, dict[str, object]] = {
    "path_exists": {"required": ("paths",), "list": ("paths",)},
    "sections_cover": {"required": ("path", "sections"), "list": ("sections",)},
    "paths_per_section": {
        "required": ("path", "sections"),
        "list": ("sections",),
        "optional": ("min_paths",),
    },
    "mentioned_paths_exist": {"required": ("path",)},
    "files_changed": {"required": ("paths",), "list": ("paths",)},
    "command_passed": {"required": ("command",)},
    "identifiers_per_section": {
        "required": ("path", "sections"),
        "list": ("sections",),
        "optional": ("min_identifiers", "categories"),
    },
    "question_types_per_section": {
        "required": ("path", "sections"),
        "list": ("sections",),
        "optional": ("question_types",),
    },
}


def validate_checks(specs: Sequence[Mapping[str, object]]) -> None:
    """校验 acceptance_checks 结构；不合法直接抛错（设置期就暴露问题）。"""

    for index, spec in enumerate(specs):
        kind = str(spec.get("kind", "")).strip()
        if kind not in CHECK_SPECS:
            raise ValueError(f"acceptance_checks[{index}]：不支持的检查器 {kind!r}")
        rule = CHECK_SPECS[kind]
        for key in rule.get("required", ()):  # type: ignore[union-attr]
            if spec.get(key) in (None, "", []):
                raise ValueError(f"acceptance_checks[{index}]（{kind}）：缺少必填参数 {key!r}")
        for key in rule.get("list", ()):  # type: ignore[union-attr]
            if not isinstance(spec.get(key), list):
                raise ValueError(f"acceptance_checks[{index}]（{kind}）：{key!r} 必须是列表")


def subsystem_criteria_text(
    subsystems: Sequence[str],
    report_path: str,
    min_identifiers: int = 3,
    min_paths: int = 2,
) -> str:
    """生成"子系统问题模板"的验收标准文本（只规定问哪几类问题，不说任何具体事实）。"""

    names = "、".join(subsystems)
    bullets = "\n".join(f"   {i}. {name}" for i, name in enumerate(QUESTION_TYPES, start=1))
    return (
        f"{report_path} 必须为每个子系统（{names}）各写一节，且每节都必须回答下面 5 类问题：\n"
        f"{bullets}\n"
        f"其中「关键数据结构」与「关键参数」两类必须写出代码里真实存在的标识符"
        f"（常量 / 类型 / 函数名，且每节至少含 {min_identifiers} 个、三类各至少 1 个），"
        f"「交互」或「关键参数」必须给出真实存在的文件路径（每节至少 {min_paths} 个）。\n"
        "报告中提到的路径与标识符必须真实存在，不得编造。"
    )


def covers_markers(spec: Mapping[str, object]) -> list[str]:
    """取一条检查声明的覆盖标记；允许写成字符串或列表。"""

    value = spec.get("covers")
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)] if value else []


def uncovered_criteria(
    criteria: str,
    specs: Sequence[Mapping[str, object]],
    results: Sequence[CheckResult] = (),
) -> tuple[str, ...]:
    """找出**没有取得脚本证据**的验收条目——它们只能被判 inconclusive。

    两种情形都算：① 没有对应的脚本检查；② 对应检查**没核验完**（unverified，
    例如标识符核验超上限）。第二种绝不能静默通过。
    """

    lines = [line.strip("- \t") for line in criteria.splitlines()]
    lines = [line for line in lines if line]
    if not specs:
        return tuple(lines)
    unverified_kinds = {r.kind for r in results if r.unverified}
    markers = [
        marker
        for spec in specs
        if str(spec.get("kind", "")) not in unverified_kinds
        for marker in covers_markers(spec)
    ]
    return tuple(line for line in lines if not any(marker in line for marker in markers))


def subsystem_checks(
    subsystems: Sequence[str],
    report_path: str,
    min_identifiers: int = 3,
    min_paths: int = 2,
    max_lookups: int = MAX_IDENTIFIER_LOOKUPS,
) -> tuple[dict[str, object], ...]:
    """把"子系统清单"翻译成结构化检查（**子系统清单由 harness 给定，不从隐藏清单推导**）。"""

    sections = list(subsystems)
    specs: list[dict[str, object]] = [
        {"kind": "sections_cover", "path": report_path, "sections": sections, "covers": "各写一节"},
        {
            "kind": "question_types_per_section",
            "path": report_path,
            "sections": sections,
            # 5 类问题的名字本身也是验收条目，必须由这条检查覆盖——
            # 否则它们会被当成"无脚本证据"而只能判 inconclusive（真机踩过）
            "covers": ["必须回答下面 5 类问题", *QUESTION_TYPES],
        },
        {
            "kind": "identifiers_per_section",
            "path": report_path,
            "sections": sections,
            "min_identifiers": min_identifiers,
            "covers": "真实存在的标识符",
            "max_lookups": max_lookups,
        },
        {
            "kind": "paths_per_section",
            "path": report_path,
            "sections": sections,
            "min_paths": min_paths,
            "covers": "真实存在的文件路径",
        },
        {"kind": "mentioned_paths_exist", "path": report_path, "covers": "不得编造"},
    ]
    return tuple(specs)
