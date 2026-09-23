"""实现 /start-skill 命令。"""

from .registry import CommandContext, SlashCommand
from .subagent import _refresh_runtime_messages


async def start_skill(context: CommandContext) -> None:
    """展示可用 skill 并让用户勾选激活。"""

    skills = context.skill_manager.list_skills()
    if not skills:
        context.screen.add_entry("tool", "No skills available")
        return
    selected = await context.screen.request_skill_picker(
        [(skill.name, skill.description, skill.source) for skill in skills],
        context.skill_manager.active_keys(),
    )
    if selected is None:
        return
    _apply_active_skills(context, selected)
    if selected:
        context.screen.set_status_message(
            f"Activated skills: {', '.join(sorted(name for name, _ in selected))}"
        )


def _apply_active_skills(context: CommandContext, selected: set[str]) -> None:
    """把用户勾选的集合写回 skill 管理器并刷新上下文注入。"""

    context.skill_manager.set_active(selected)
    _refresh_runtime_messages(context)


start_skill_command = SlashCommand(
    name="start-skill",
    description="Select and activate skills",
    handler=start_skill,
)
