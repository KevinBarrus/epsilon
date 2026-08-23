"""slash command 注册表与内置命令。"""

from .registry import (
    CommandContext,
    CommandRegistrationError,
    CommandRegistry,
    SlashCommand,
)
from .start_skill import start_skill_command
from .stop_skill import stop_skill_command
from .model import model_command_slash
from .thinking import thinking_command_slash
from .skills import skills_command_slash
from .mcp import mcp_command_slash
from .compact import compact_command_slash
from .status import status_command_slash
from .copy import copy_command_slash
from .clear import clear_command_slash
from .quit import quit_command_slash
from .export import export_command_slash
from .diff import diff_command_slash
from .background_image import background_image_command_slash
from .delete import delete_command_slash
from .thinking_toggle import thinking_toggle_command_slash

__all__ = [
    "CommandContext",
    "CommandRegistrationError",
    "CommandRegistry",
    "SlashCommand",
    "start_skill_command",
    "stop_skill_command",
    "model_command_slash",
    "thinking_command_slash",
    "skills_command_slash",
    "mcp_command_slash",
    "compact_command_slash",
    "status_command_slash",
    "copy_command_slash",
    "clear_command_slash",
    "quit_command_slash",
    "export_command_slash",
    "diff_command_slash",
    "background_image_command_slash",
    "delete_command_slash",
    "thinking_toggle_command_slash",
]
