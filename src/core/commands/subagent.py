"""实现 /subagent 命令：查看并切换 Scout。"""

from ..subagent import scout_parent_message
from .registry import CommandContext, SlashCommand


async def subagent_command(context: CommandContext) -> None:
    """展示 Scout 状态并在当前进程内切换开关。"""

    if context.tool_manager is None:
        context.screen.add_entry("tool", "Scout unavailable")
        return
    current = context.tool_manager.is_model_tool_enabled("spawn_agent")
    choice = await context.screen.request_choice_picker(
        ["on", "off"],
        f"Scout is {'on' if current else 'off'} (↑/↓ move, Enter confirm, Esc cancel)",
    )
    if choice is not None:
        context.tool_manager.set_model_tool_enabled(
            "spawn_agent",
            choice == "on",
        )
        context.context_manager.update_model_tools(
            context.tool_manager.model_tools()
        )
        _refresh_runtime_messages(context)
        current = choice == "on"
    context.screen.add_entry(
        "tool",
        "\n".join(
            (
                f"Scout: {'on' if current else 'off'}",
                f"model: {context.client_holder.settings.model_name} (inherits main)",
                f"thinking level: {context.agent_loop.thinking_level} (inherits main)",
            )
        ),
    )


def _refresh_runtime_messages(context: CommandContext) -> None:
    """合并当前 Skill 与 Scout 说明，避免两个运行时开关互相覆盖。"""

    messages = context.skill_manager.active_system_messages()
    if (
        context.tool_manager is not None
        and context.tool_manager.is_model_tool_enabled("spawn_agent")
    ):
        messages.append(scout_parent_message())
    context.context_manager.set_extra_system_messages(messages)


subagent_command_slash = SlashCommand(
    name="subagent",
    description="Show or toggle the read-only Scout",
    handler=subagent_command,
)
