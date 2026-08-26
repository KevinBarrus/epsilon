"""定义 Agent 自然结束前的可选收尾策略。"""

from dataclasses import dataclass
import shlex
from collections.abc import Sequence
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
DIRECT_VERIFICATION_COMMANDS = frozenset({"pytest", "tox", "nox", "flake8", "mypy", "eslint"})
PYTHON_MODULE_CHECKS = frozenset({"pytest", "unittest", "compileall", "py_compile"})
TEST_SUBCOMMANDS = frozenset({"make", "cargo", "go", "npm", "pnpm", "yarn"})
SHELL_CONNECTORS = frozenset({"&&", "||", ";", "|"})


@dataclass(frozen=True)
class EndPolicySummary:
    """保存收尾策略可追溯的最小运行结果。"""

    verification_reminder_injected: bool
    write_count: int
    post_write_command_results: tuple[ToolResult, ...]
    verification_command_results: tuple[ToolResult, ...]


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
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    return any(_is_verification_segment(segment) for segment in _command_segments(tokens))


def _command_segments(tokens: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """按 shell 连接符拆分命令，避免把参数文本当作命令名。"""

    segments: list[tuple[str, ...]] = []
    current: list[str] = []
    for token in tokens:
        if token in SHELL_CONNECTORS:
            if current:
                segments.append(tuple(current))
                current = []
        else:
            current.append(token)
    if current:
        segments.append(tuple(current))
    return tuple(segments)


def _is_verification_segment(tokens: Sequence[str]) -> bool:
    """只按命令名及独立参数判断单段 shell 命令。"""

    if not tokens:
        return False
    command = tokens[0].rsplit("/", 1)[-1]
    if command in DIRECT_VERIFICATION_COMMANDS:
        return True
    if command in {"python", "python3", "python3.11"} and len(tokens) >= 3:
        return (
            tokens[1] == "-m" and tokens[2] in PYTHON_MODULE_CHECKS
            or tokens[1].rsplit("/", 1)[-1] == "manage.py" and tokens[2] == "test"
        )
    if command == "manage.py" and len(tokens) >= 2:
        return tokens[1] == "test"
    if command == "ruff" and len(tokens) >= 2:
        return tokens[1] == "check"
    if command in TEST_SUBCOMMANDS and len(tokens) >= 2:
        return tokens[1] == "test" or tokens[1] == "run" and len(tokens) >= 3 and tokens[2] == "test"
    if command in {"uv", "poetry"} and len(tokens) >= 3:
        return tokens[1] == "run" and _is_verification_segment(tokens[2:])
    return False


class WriteVerificationPolicy:
    """要求成功写入文件后至少尝试一次命令验证。"""

    def __init__(self) -> None:
        """初始化本轮独立的写入与验证状态。"""

        self._write_count = 0
        self._needs_verification = False
        self._last_verification_failed = False
        self._reminder_injected = False
        self._post_write_command_results: list[ToolResult] = []
        self._verification_command_results: list[ToolResult] = []

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
                if is_verification_command(tool_call):
                    self._verification_command_results.append(result)
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
            tuple(self._verification_command_results),
        )
