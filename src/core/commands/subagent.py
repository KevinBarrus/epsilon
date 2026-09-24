"""实现 /subagent 命令：查看并切换 Scout。"""

from ..subagent import subagent_parent_message
from ..goal import GoalPolicy
from .registry import CommandContext, SlashCommand

SUBAGENT_TOOL_NAMES = ("spawn_agent", "spawn_worker", "spawn_reviewer")


async def subagent_command(context: CommandContext) -> None:
    """展示三种子 Agent 状态，并统一切换工具可见性。"""

    if context.tool_manager is None:
        context.screen.add_entry("tool", "Scout unavailable")
        return
    states = {
        name: context.tool_manager.is_model_tool_enabled(name)
        for name in SUBAGENT_TOOL_NAMES
    }
    current = (
        "on" if all(states.values()) else "off" if not any(states.values()) else "mixed"
    )
    choice = await context.screen.request_choice_picker(
        ["on", "off"],
        f"Subagents are {current} "
        "(Scout / Worker / Reviewer; ↑/↓ move, Enter confirm, Esc cancel)",
    )
    if choice is not None:
        for name in SUBAGENT_TOOL_NAMES:
            context.tool_manager.set_model_tool_enabled(name, choice == "on")
        context.context_manager.update_model_tools(
            context.tool_manager.model_tools()
        )
        _refresh_runtime_messages(context)
        states = {name: choice == "on" for name in SUBAGENT_TOOL_NAMES}
    context.screen.add_entry(
        "tool",
        "\n".join(
            (
                f"Scout: {'on' if states['spawn_agent'] else 'off'}",
                f"Worker: {'on' if states['spawn_worker'] else 'off'}",
                f"Reviewer: {'on' if states['spawn_reviewer'] else 'off'}",
                f"model: {context.client_holder.settings.model_name} (inherits main)",
                f"thinking level: {context.agent_loop.thinking_level} (inherits main)",
            )
        ),
    )


def _refresh_runtime_messages(context: CommandContext) -> None:
    """合并当前 Skill 与 Scout 说明，避免两个运行时开关互相覆盖。"""

    messages = context.skill_manager.active_system_messages()
    policy = getattr(context.agent_loop, "end_policy", None)
    if isinstance(policy, GoalPolicy):
        messages.append(policy.instruction_message())
    if context.tool_manager is not None and any(
        context.tool_manager.is_model_tool_enabled(name)
        for name in SUBAGENT_TOOL_NAMES
    ):
        messages.append(subagent_parent_message())
    context.context_manager.set_extra_system_messages(messages)


subagent_command_slash = SlashCommand(
    name="subagent",
    description="Show or toggle Scout, Worker, and Reviewer",
    handler=subagent_command,
)
