"""实现 /goal 的设置、查看、清除与恢复时运行态同步。"""

from ..goal import Goal, GoalPolicy
from .registry import CommandContext, SlashCommand
from .subagent import _refresh_runtime_messages


def sync_goal_runtime(context: CommandContext) -> None:
    """按 Session 快照同步结束策略、目标工具与系统说明。"""
    goal = context.session.get_goal()
    if goal is not None and goal.status == "active":
        policy = GoalPolicy(goal, on_change=lambda _: context.session.update_goal())
        context.agent_loop.set_end_policy(policy)
    else:
        if isinstance(getattr(context.agent_loop, "end_policy", None), GoalPolicy):
            context.agent_loop.set_end_policy(None)
    if context.tool_manager is not None:
        context.tool_manager.set_model_tool_enabled("goal", goal is not None and goal.status == "active")
        context.context_manager.update_model_tools(context.tool_manager.model_tools())
    _refresh_runtime_messages(context)


async def goal_command(context: CommandContext) -> None:
    """处理 /goal <objective>、/goal status 与 /goal clear。"""
    argument = context.arguments.strip()
    if not argument or argument == "status":
        goal = context.session.get_goal()
        if goal is None:
            context.screen.add_entry("tool", "当前没有目标。用 /goal <目标> 设置。")
        else:
            context.screen.add_entry(
                "tool",
                f"目标：{goal.objective}\n状态：{goal.status}\n"
                f"轮次：{goal.rounds_started}/{goal.max_rounds if goal.max_rounds is not None else '不限'}\n"
                f"token：{goal.tokens_used}/{goal.token_budget if goal.token_budget is not None else '不限'}\n"
                f"时间：{goal.elapsed_seconds:.0f}/{goal.time_budget_seconds if goal.time_budget_seconds is not None else '不限'} 秒",
            )
        return
    if argument == "clear":
        context.session.clear_goal()
        sync_goal_runtime(context)
        context.screen.add_entry("tool", "目标已清除。")
        return
    context.session.set_goal(Goal(argument, max_rounds=50, token_budget=5_000_000))
    sync_goal_runtime(context)
    context.screen.add_entry("tool", f"已设置持续目标：{argument}")


goal_command_slash = SlashCommand(
    name="goal",
    description="Set, inspect, or clear a persistent goal",
    handler=goal_command,
)
