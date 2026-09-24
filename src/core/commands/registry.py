"""统一注册和分发 slash command。"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path

from ..agent_loop import AgentLoop
from ..context import ContextManager
from ..model import ClientHolder
from ..screen import ChatScreen
from ..session import Session
from ..skills import SkillManager


class CommandRegistrationError(ValueError):
    """命令重复注册时抛出的异常。"""


@dataclass(frozen=True)
class CommandContext:
    """一次命令执行可用的应用依赖。"""

    screen: ChatScreen
    session: Session
    skill_manager: SkillManager
    context_manager: ContextManager
    client_holder: ClientHolder
    agent_loop: AgentLoop
    project_dir: Path
    tool_manager: "ToolManager | None" = None
    arguments: str = ""


@dataclass(frozen=True)
class SlashCommand:
    """描述一个可注册的 slash command。"""

    name: str
    description: str
    handler: Callable[[CommandContext], Awaitable[None]]


class CommandRegistry:
    """保存 slash command 并提供统一分发入口。"""

    def __init__(self) -> None:
        """创建空的命令注册表。"""

        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        """注册一个命令，重复名称直接拒绝。"""

        if command.name in self._commands:
            raise CommandRegistrationError(f"Command already registered: {command.name}")
        self._commands[command.name] = command

    def get(self, name: str) -> SlashCommand | None:
        """根据名称查找命令。"""

        return self._commands.get(name)

    def list(self) -> list[SlashCommand]:
        """返回全部已注册命令。"""

        return list(self._commands.values())

    async def dispatch(self, line: str, context: CommandContext) -> bool:
        """解析 /name 并分发到对应处理器，命中返回 True。"""

        if not line.startswith("/"):
            return False
        name = line[1:].split(None, 1)[0] if len(line) > 1 else ""
        command = self._commands.get(name)
        if command is None:
            return False
        arguments = line[1:].partition(" ")[2].strip()
        await command.handler(
            replace(context, arguments=arguments)
            if isinstance(context, CommandContext)
            else context
        )
        return True
