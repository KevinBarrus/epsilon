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
    """一条脚本化检查的结果。"""

    kind: str
    passed: bool
    detail: str = ""


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
            lines = [f"[{'PASS' if r.passed else 'FAIL'}] {r.kind}: {r.detail}" for r in self.checks]
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
