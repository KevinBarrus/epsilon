"""定义 Agent 自然结束前的可选收尾策略。"""

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Protocol

from .model import Message, ToolCall, ToolResult


VERIFICATION_REMINDER = (
    "You modified files but have not run a verification command afterwards. "
    "Run the most relevant check now, or clearly explain why verification cannot run."
)
FAILED_VERIFICATION_REMINDER = (
    "Your verification command failed after modifying files. Read its output, fix the issue "
    "and rerun the relevant check, or clearly explain the blocker."
)
VERIFICATION_COMMAND_PATTERNS = (
    re.compile(r"\bpytest\b"),
    re.compile(r"\bpython(?:\d(?:\.\d+)?)?\s+-m\s+unittest\b"),
    re.compile(r"\b(?:python(?:\d(?:\.\d+)?)?\s+)?manage\.py\s+test\b"),
    re.compile(r"\b(?:tox|nox)\b"),
    re.compile(r"\b(?:python(?:\d(?:\.\d+)?)?\s+-m\s+)?(?:compileall|py_compile)\b"),
    re.compile(r"\b(?:ruff\s+check|flake8|mypy|eslint)\b"),
    re.compile(r"\b(?:make|cargo|go|npm|pnpm|yarn)\s+test\b"),
)


@dataclass(frozen=True)
class EndPolicySummary:
    """保存收尾策略可追溯的最小运行结果。"""

    verification_reminder_injected: bool
    write_count: int
    post_write_command_results: tuple[ToolResult, ...]


class TurnEndPolicy(Protocol):
    """定义 Agent 在模型自然结束前可执行的最小决策接口。"""

    def observe_tool_results(
        self,
        tool_calls: Sequence[ToolCall],
        results: Sequence[ToolResult],
    ) -> None:
        """接收一批按模型调用顺序排列的工具结果。"""

    def follow_up_message(self) -> Message | None:
        """返回至多一次的后续上下文消息，或允许自然结束。"""

    @property
    def summary(self) -> EndPolicySummary:
        """返回本轮收尾决策的可追溯结果。"""


def is_verification_command(tool_call: ToolCall) -> bool:
    """判断命令工具调用是否具有测试、检查或构建验证语义。"""

    command = tool_call.arguments.get("command")
    if not isinstance(command, str):
        return False
    return any(pattern.search(command) for pattern in VERIFICATION_COMMAND_PATTERNS)


class WriteVerificationPolicy:
    """要求成功写入文件后至少尝试一次命令验证。"""

    def __init__(self) -> None:
        """初始化本轮独立的写入与验证状态。"""

        self._write_count = 0
        self._needs_verification = False
        self._last_verification_failed = False
        self._reminder_injected = False
        self._post_write_command_results: list[ToolResult] = []

    def observe_tool_results(
        self,
        tool_calls: Sequence[ToolCall],
        results: Sequence[ToolResult],
    ) -> None:
        """记录成功写入以及其后发生的命令验证尝试。"""

        for tool_call, result in zip(tool_calls, results):
            if tool_call.name in {"write_file", "edit_file"} and not result.is_error:
                self._write_count += 1
                self._needs_verification = True
                self._last_verification_failed = False
            elif tool_call.name == "run_command" and self._needs_verification:
                self._post_write_command_results.append(result)
                self._needs_verification = result.is_error
                self._last_verification_failed = result.is_error

    def follow_up_message(self) -> Message | None:
        """在缺少或失败的写后验证时最多追加一次固定提醒。"""

        if not self._needs_verification or self._reminder_injected:
            return None
        self._reminder_injected = True
        reminder = (
            FAILED_VERIFICATION_REMINDER
            if self._last_verification_failed
            else VERIFICATION_REMINDER
        )
        return Message(role="system", content=reminder)

    @property
    def summary(self) -> EndPolicySummary:
        """返回当前轮次的写入、验证和提醒状态。"""

        return EndPolicySummary(
            self._reminder_injected,
            self._write_count,
            tuple(self._post_write_command_results),
        )
