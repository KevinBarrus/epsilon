"""Agent 空转检测与纠偏（Loop Guard v2）。

把"空转"当信号而不是时长问题：检测到局部循环后注入一条系统消息把模型拽出来，
不设置任何轮次或时间上限（生产长任务不能被掐死）。

v2 相对 v1 的三处改造：
- 检测 A 补两种信号：`abab_action_cycle`（A,B,A,B 交替循环）与
  `same_error_family`（同一错误族反复出现，换调用也抓）；
- 检测 B 改为"是否产生新事实"：只有"原地打转、读取集合不再增长"才报警，
  合法的探索期不再误报；
- 提醒节制：**每轮最多注入一条**，多条候选按
  `进展不变 > 错误族 > 重复调用 > abab` 取最高优先级。

判定入口下沉到 `ActionFact`（一轮里每个调用的已提取事实），实时观察与离线回放
共用同一套逻辑，保证回放结论与真机一致。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .model import Message, ToolCall, ToolResult

if TYPE_CHECKING:  # 仅用于类型标注，避免 config → loop_guard → model 的循环导入
    from .config import Settings

# 达到最后一个阈值后，每隔多少轮继续施压一次（各信号共用节流口径）
REPEAT_REMINDER_INTERVAL = 4
# 只读角色天然没有写入，跳过"无新事实"检测，避免误报
READ_ONLY_ROLES = frozenset({"scout"})
# 参数与结果摘要长度上限（人读用）
EXCERPT_LIMIT = 200
ARGS_PREVIEW_LIMIT = 120
# abab 观察窗口：最近 4 轮呈 A,B,A,B
ABAB_WINDOW = 4

SignalKind = Literal[
    "no_progress",
    "same_error_family",
    "repeated_call",
    "abab_action_cycle",
]
ReminderLevel = Literal["mild", "detailed"]

# 每轮最多注入一条提醒，多条候选按此优先级挑最要紧的
SIGNAL_PRIORITY: tuple[SignalKind, ...] = (
    "no_progress",
    "same_error_family",
    "repeated_call",
    "abab_action_cycle",
)

# 纠偏消息必须声明临时性，避免模型把它当成用户偏好写进持久记忆
ANTI_PERSISTENCE = (
    "这是本轮临时提醒，不是用户偏好；不要把它写进 Memory / Skills 或任何持久文件。"
)

REPEATED_CALL_REMINDER = (
    "你在用完全相同的参数重复调用同一个工具。先仔细分析上一次的结果："
    "如果任务还没完成，请换一种方法或换参数，不要重复同样的调用。"
)


def canonical_arguments(arguments: Mapping[str, object]) -> str:
    """把工具参数规范化成稳定字符串，作为签名与摘要的共同口径。"""

    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)


def action_digest(name: str, arguments: Mapping[str, object]) -> str:
    """返回"工具名 + 规范化参数"的短摘要，供检测与观测归因共用。"""

    payload = f"{name}:{canonical_arguments(arguments)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def args_preview(arguments: Mapping[str, object], limit: int = ARGS_PREVIEW_LIMIT) -> str:
    """返回便于人读的参数预览（截断）。"""

    return _excerpt(canonical_arguments(arguments), limit)


def output_fingerprint(content: str) -> str:
    """返回一次工具输出的指纹，用于判断"命令产生了新输出"。"""

    return hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:16]


def error_family(is_error: bool, error_category: str | None, content: str) -> str | None:
    """返回错误族：优先结构化错误码，无码时取规范化后的错误首行。

    参数用原始字段而不是 ToolResult，便于离线回放复用同一口径。
    """

    if not is_error:
        return None
    if error_category:
        return error_category
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    if not lines:
        return "unknown_error"
    return _excerpt(lines[0], 80)


def _excerpt(text: str, limit: int = EXCERPT_LIMIT) -> str:
    """截断长文本，避免纠偏消息本身撑大上下文。"""

    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "…"


def _round_fingerprint(digests: Sequence[str]) -> str:
    """把一个批次的所有调用签名排序后压成一个轮指纹。"""

    payload = ";".join(sorted(digests))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ActionFact:
    """一轮里单个工具调用的已提取事实。

    字段与归档记录一一对应，使离线回放可以复用完全相同的判定逻辑。
    """

    tool: str
    args_digest: str
    args_preview: str = ""
    is_error: bool = False
    error_family: str | None = None
    output_fingerprint: str | None = None
    result_preview: str = ""


@dataclass(frozen=True)
class LoopGuardConfig:
    """Loop Guard 的可配置项。"""

    enabled: bool = True
    thresholds: tuple[int, ...] = (3, 5, 8)
    exempt_tools: tuple[str, ...] = ()
    no_progress_rounds: int = 8

    def __post_init__(self) -> None:
        """校验阈值为非空递增正整数，无进展轮数非负。"""

        if not self.thresholds:
            raise ValueError("thresholds must not be empty")
        previous = 0
        for threshold in self.thresholds:
            if threshold <= previous:
                raise ValueError("thresholds must be strictly increasing positive ints")
            previous = threshold
        if self.no_progress_rounds < 0:
            raise ValueError("no_progress_rounds must be >= 0")


def config_from_settings(settings: "Settings") -> LoopGuardConfig:
    """从运行配置构造空转检测配置。"""

    return LoopGuardConfig(
        enabled=settings.loop_guard_enabled,
        thresholds=settings.loop_guard_thresholds,
        exempt_tools=settings.loop_guard_exempt_tools,
        no_progress_rounds=settings.loop_guard_no_progress_rounds,
    )


@dataclass(frozen=True)
class LoopGuardEvent:
    """一次纠偏注入的可观测记录。"""

    kind: SignalKind
    level: ReminderLevel
    tool_name: str | None
    count: int
    agent_role: str = ""
    agent_run_id: str = ""


@dataclass(frozen=True)
class LoopGuardInjection:
    """一次纠偏注入：注入的消息与对应事件。"""

    message: Message
    event: LoopGuardEvent


def _detailed_repeat_message(
    tool_name: str,
    count: int,
    arguments: str,
    last_result: str,
) -> str:
    """渲染重复调用的详细提醒。"""

    lines = [
        "检测到重复工具调用：",
        f"- 工具：{tool_name}",
        f"- 连续次数：{count}",
        f"- 参数：{_excerpt(arguments)}",
    ]
    if last_result:
        lines.append(f"- 上次结果：{_excerpt(last_result)}")
    lines.append("不要继续重复；换策略，或明确说明当前阻塞。")
    return "\n".join(lines)


def _no_progress_message(count: int) -> str:
    """渲染"原地打转、没有新事实"的提醒。"""

    return (
        f"你已经连续 {count} 轮没有产生任何新事实：没有新的动作、没有成功的写入或委派、"
        "命令输出也没有变化。你在原地打转。重新评估：是卡住了、缺信息，还是该换个方案？"
        "如果确实无法推进，请说明阻塞原因。"
    )


def _error_family_message(family: str, count: int, last_result: str) -> str:
    """渲染同一错误族反复出现的提醒。"""

    lines = [
        "检测到同一类错误反复出现：",
        f"- 错误族：{family}",
        f"- 连续次数：{count}",
    ]
    if last_result:
        lines.append(f"- 最近结果：{_excerpt(last_result)}")
    lines.append("不要再用同样的方式重试；换思路，或明确说明卡在哪里。")
    return "\n".join(lines)


def _abab_message(count: int) -> str:
    """渲染 A,B,A,B 交替循环的提醒。"""

    return (
        "检测到动作在两种批次之间来回交替（A,B,A,B）——例如「改回去再跑一次」这类循环，"
        "整体没有推进。请停下来判断：这两步里哪一步是无效的？能不能直接做正确的那一步？\n"
        f"- 已持续：{count} 个观察窗口"
    )


class ToolCallLoopGuard:
    """检测单个 Agent 的空转并给出纠偏消息。

    每个 AgentLoop（含每个子 Agent）各持一个实例，计数互不影响。
    """

    def __init__(self, config: LoopGuardConfig | None = None, role: str = "parent") -> None:
        """创建 guard，记录角色以决定是否启用"无新事实"检测。"""

        self._config = config or LoopGuardConfig()
        self._role = role
        self._repeat_counts: dict[str, int] = {}
        self._repeat_meta: dict[str, tuple[str, str]] = {}
        self._repeat_results: dict[str, str] = {}
        self._seen_facts: set[str] = set()
        self._command_outputs: dict[str, set[str]] = {}
        self._rounds_without_fact = 0
        self._error_counts: dict[str, int] = {}
        self._error_results: dict[str, str] = {}
        self._round_fingerprints: list[str] = []
        self._abab_rounds = 0
        self._injections = 0

    @property
    def injections(self) -> int:
        """返回累计注入次数。"""

        return self._injections

    @property
    def rounds_without_new_fact(self) -> int:
        """返回当前连续"无新事实"的轮数，供观测与回放使用。"""

        return self._rounds_without_fact

    def observe(
        self,
        tool_calls: Sequence[ToolCall],
        results: Sequence[ToolResult],
        capabilities: Mapping[str, str | None],
    ) -> LoopGuardInjection | None:
        """观察一轮工具结果，命中空转时返回至多一条纠偏注入。"""

        facts = tuple(
            ActionFact(
                tool=call.name,
                args_digest=action_digest(call.name, call.arguments),
                args_preview=args_preview(call.arguments),
                is_error=result.is_error,
                error_family=error_family(
                    result.is_error,
                    result.error_category,
                    result.content,
                ),
                output_fingerprint=(
                    output_fingerprint(result.content)
                    if call.name == "run_command" and not result.is_error
                    else None
                ),
                result_preview=_excerpt(result.content),
            )
            for call, result in zip(tool_calls, results)
        )
        return self.observe_facts(facts, capabilities)

    def observe_facts(
        self,
        facts: Sequence[ActionFact],
        capabilities: Mapping[str, str | None],
    ) -> LoopGuardInjection | None:
        """用已提取的事实执行检测，实时观察与离线回放共用此入口。"""

        if not self._config.enabled:
            return None
        round_facts = tuple(facts)
        self._observe_new_facts(round_facts, capabilities)
        self._update_repeat_counts(round_facts)
        self._update_error_families(round_facts)
        self._update_abab(round_facts)

        for kind in SIGNAL_PRIORITY:
            injection = self._build_injection(kind)
            if injection is not None:
                return self._emit(injection)
        return None

    def _observe_new_facts(
        self,
        facts: tuple[ActionFact, ...],
        capabilities: Mapping[str, str | None],
    ) -> bool:
        """判断本轮是否产生新事实，并把本轮签名纳入已见集合。"""

        new_fact = False
        for fact in facts:
            capability = capabilities.get(fact.tool) or ""
            first_seen = fact.args_digest not in self._seen_facts
            fingerprint = fact.output_fingerprint

            if first_seen:
                new_fact = True
            elif not fact.is_error and (
                capability == "file.write" or capability.startswith("agent.")
            ):
                # 成功的写入 / 委派本身就是新事实
                new_fact = True
            elif fingerprint is not None:
                # 同命令但输出指纹没出现过，也算新事实
                if fingerprint not in self._command_outputs.setdefault(fact.args_digest, set()):
                    new_fact = True

            if fingerprint is not None:
                # 首次出现也要登记，否则第二次会被误判为新输出
                self._command_outputs.setdefault(fact.args_digest, set()).add(fingerprint)
            self._seen_facts.add(fact.args_digest)

        if new_fact:
            self._rounds_without_fact = 0
        else:
            self._rounds_without_fact += 1
        return new_fact

    def _update_repeat_counts(self, facts: tuple[ActionFact, ...]) -> None:
        """按签名更新连续重复计数，本轮未出现的签名归零。"""

        present: dict[str, ActionFact] = {}
        for fact in facts:
            if fact.tool in self._config.exempt_tools:
                continue
            present[fact.args_digest] = fact

        for digest in list(self._repeat_counts):
            if digest not in present:
                self._repeat_counts.pop(digest, None)
                self._repeat_meta.pop(digest, None)
                self._repeat_results.pop(digest, None)
        for digest, fact in present.items():
            self._repeat_counts[digest] = self._repeat_counts.get(digest, 0) + 1
            self._repeat_meta[digest] = (fact.tool, fact.args_preview)
            self._repeat_results[digest] = fact.result_preview

    def _update_error_families(self, facts: tuple[ActionFact, ...]) -> None:
        """按错误族更新连续计数，本轮未出现的族归零。"""

        present: dict[str, str] = {}
        for fact in facts:
            if fact.error_family is not None:
                present[fact.error_family] = fact.result_preview
        for family in list(self._error_counts):
            if family not in present:
                self._error_counts.pop(family, None)
                self._error_results.pop(family, None)
        for family, preview in present.items():
            self._error_counts[family] = self._error_counts.get(family, 0) + 1
            self._error_results[family] = preview

    def _update_abab(self, facts: tuple[ActionFact, ...]) -> None:
        """维护轮指纹窗口，识别 A,B,A,B 交替循环。"""

        self._round_fingerprints.append(
            _round_fingerprint([fact.args_digest for fact in facts])
        )
        self._round_fingerprints = self._round_fingerprints[-ABAB_WINDOW:]
        window = self._round_fingerprints
        alternates = (
            len(window) == ABAB_WINDOW
            and window[0] == window[2]
            and window[1] == window[3]
            and window[0] != window[1]
        )
        self._abab_rounds = self._abab_rounds + 1 if alternates else 0

    def _build_injection(self, kind: SignalKind) -> LoopGuardInjection | None:
        """按信号类型构造注入，未命中返回 None。"""

        if kind == "repeated_call":
            return self._repeated_call_injection()
        if kind == "no_progress":
            return self._no_progress_injection()
        if kind == "same_error_family":
            return self._error_family_injection()
        return self._abab_injection()

    def _repeated_call_injection(self) -> LoopGuardInjection | None:
        """返回计数最高且达到阈值的重复调用提醒。"""

        best: tuple[int, str, ReminderLevel] | None = None
        for digest, count in self._repeat_counts.items():
            level = self._repeat_level(count)
            if level is None:
                continue
            if best is None or count > best[0]:
                best = (count, digest, level)
        if best is None:
            return None
        count, digest, level = best
        tool_name, arguments = self._repeat_meta[digest]
        if level == "mild":
            content = REPEATED_CALL_REMINDER
        else:
            content = _detailed_repeat_message(
                tool_name,
                count,
                arguments,
                self._repeat_results.get(digest, ""),
            )
        return LoopGuardInjection(
            Message(role="system", content=f"{content}\n{ANTI_PERSISTENCE}"),
            LoopGuardEvent("repeated_call", level, tool_name, count),
        )

    def _repeat_level(self, count: int) -> ReminderLevel | None:
        """按阈值决定提醒级别：命中阈值为温和/详细，之后每 4 轮继续施压。"""

        thresholds = self._config.thresholds
        if count in thresholds:
            return "mild" if count == thresholds[0] else "detailed"
        last = thresholds[-1]
        if count > last and (count - last) % REPEAT_REMINDER_INTERVAL == 0:
            return "detailed"
        return None

    def _no_progress_injection(self) -> LoopGuardInjection | None:
        """连续无新事实达到阈值时返回提醒，之后按节流继续施压。"""

        limit = self._config.no_progress_rounds
        if limit == 0 or self._role in READ_ONLY_ROLES:
            return None
        count = self._rounds_without_fact
        if count < limit:
            return None
        if count != limit and (count - limit) % REPEAT_REMINDER_INTERVAL != 0:
            return None
        return LoopGuardInjection(
            Message(role="system", content=f"{_no_progress_message(count)}\n{ANTI_PERSISTENCE}"),
            LoopGuardEvent("no_progress", "detailed", None, count),
        )

    def _error_family_injection(self) -> LoopGuardInjection | None:
        """同一错误族连续达到阈值时返回提醒。"""

        best: tuple[int, str, ReminderLevel] | None = None
        for family, count in self._error_counts.items():
            level = self._repeat_level(count)
            if level is None:
                continue
            if best is None or count > best[0]:
                best = (count, family, level)
        if best is None:
            return None
        count, family, level = best
        message = _error_family_message(family, count, self._error_results.get(family, ""))
        return LoopGuardInjection(
            Message(role="system", content=f"{message}\n{ANTI_PERSISTENCE}"),
            LoopGuardEvent("same_error_family", level, None, count),
        )

    def _abab_injection(self) -> LoopGuardInjection | None:
        """识别到 A,B,A,B 交替循环并按节流返回提醒。"""

        level = self._repeat_level(self._abab_rounds)
        if level is None:
            return None
        return LoopGuardInjection(
            Message(role="system", content=f"{_abab_message(self._abab_rounds)}\n{ANTI_PERSISTENCE}"),
            LoopGuardEvent("abab_action_cycle", level, None, self._abab_rounds),
        )

    def _emit(self, injection: LoopGuardInjection) -> LoopGuardInjection:
        """记录注入次数并返回该注入。"""

        self._injections += 1
        return injection
