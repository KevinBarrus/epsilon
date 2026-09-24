"""测试持久目标的续跑、预算、工具与会话恢复。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agent_loop import AgentLoop
from core.commands.goal import goal_command_slash, sync_goal_runtime
from core.commands.registry import CommandContext, CommandRegistry
from core.goal import Goal, GoalPolicy, create_goal_tool
from core.model import Message, TextDelta, ToolCall, ToolCallEvent, UsageEvent
from core.session import Session
from core.tools import ToolManager


def test_goal_policy_continues_until_explicit_completion() -> None:
    """模型自然停下不能把未完成目标改成小任务。"""
    goal = Goal("完成全部 73 个模块", max_rounds=3)
    policy = GoalPolicy(goal)

    message = policy.follow_up_message()

    assert message is not None
    assert "绝不允许把成功重新定义成一个更小" in message.content
    assert goal.rounds_started == 1
    assert policy.mark_complete()
    assert goal.status == "complete"
    assert policy.follow_up_message() is None


def test_goal_policy_limits_rounds_tokens_and_time() -> None:
    """任一预算耗尽只注入一次无工具收尾消息。"""
    for goal in (
        Goal("任务", max_rounds=0),
        Goal("任务", token_budget=10),
        Goal("任务", time_budget_seconds=1),
    ):
        now = [0.0]
        policy = GoalPolicy(goal, clock=lambda: now[0])
        if goal.token_budget is not None:
            policy.observe_usage(UsageEvent(8, 2, 10))
        if goal.time_budget_seconds is not None:
            now[0] = 2.0
        close = policy.follow_up_message()
        assert close is not None and "预算" in close.content
        assert goal.status == "budget_limited"
        assert policy.final_response_only
        assert policy.follow_up_message() is None


@pytest.mark.asyncio
async def test_goal_tool_checks_completion_and_requires_no_approval() -> None:
    """硬验证未过时不得标记完成；通过后工具可直接更新目标。"""
    verified = [False]
    policy = GoalPolicy(Goal("完成迁移"), completion_check=lambda: verified[0])
    manager = ToolManager()
    manager.register_local(*create_goal_tool(lambda: policy))

    rejected = await manager.execute(ToolCall("one", "goal", {"op": "complete"}))
    assert rejected.is_error
    assert policy.goal.status == "active"

    verified[0] = True
    accepted = await manager.execute(ToolCall("two", "goal", {"op": "complete"}))
    assert not accepted.is_error
    assert policy.goal.status == "complete"


def test_goal_jsonl_round_trip_and_clear(tmp_path: Path) -> None:
    """目标快照与清除记录可跨 Session 恢复，消息读取忽略目标记录。"""
    session = Session(tmp_path)
    goal = Goal("迁移全部模块", max_rounds=50, token_budget=5_000_000)
    session.set_goal(goal)
    goal.tokens_used = 321
    goal.rounds_started = 2
    session.update_goal()
    session.add_user_message("开始")
    assert session.flush_persistence()
    session_id = session.session_id
    session.close()

    restored = Session.restore(tmp_path, session_id)
    assert restored.get_goal() == goal
    assert restored.get_messages() == [Message(role="user", content="开始")]
    restored.clear_goal()
    restored.close()
    assert Session.restore(tmp_path, session_id).get_goal() is None
    records = [json.loads(line) for line in (tmp_path / ".epsilon/sessions" / f"{session_id}.jsonl").read_text().splitlines()]
    assert [record["type"] for record in records].count("goal") == 3


@pytest.mark.asyncio
async def test_goal_command_sets_shows_clears_and_restores_runtime(tmp_path: Path) -> None:
    """/goal 参数通过分发器传入，恢复后重新启用策略与工具。"""
    session = Session(tmp_path)
    manager = ToolManager()
    agent = AgentLoop(SimpleNamespace(), manager)
    manager.register_local(*create_goal_tool(lambda: agent.end_policy))
    manager.set_model_tool_enabled("goal", False)
    entries = []
    context_updates = []
    runtime_messages = []
    context = CommandContext(
        screen=SimpleNamespace(add_entry=lambda role, content: entries.append(content)),
        session=session,
        skill_manager=SimpleNamespace(active_system_messages=lambda: []),
        context_manager=SimpleNamespace(
            update_model_tools=context_updates.append,
            set_extra_system_messages=runtime_messages.append,
        ),
        client_holder=SimpleNamespace(),
        agent_loop=agent,
        project_dir=tmp_path,
        tool_manager=manager,
    )
    registry = CommandRegistry()
    registry.register(goal_command_slash)

    assert await registry.dispatch("/goal 迁移全部模块", context)
    assert session.get_goal().objective == "迁移全部模块"
    assert session.get_goal().max_rounds == 50
    assert session.get_goal().token_budget == 5_000_000
    assert manager.is_model_tool_enabled("goal")
    assert isinstance(agent.end_policy, GoalPolicy)
    assert "迁移全部模块" in runtime_messages[-1][-1].content

    await registry.dispatch("/goal status", context)
    assert "active" in entries[-1]
    session_id = session.session_id
    session.close()
    restored = Session.restore(tmp_path, session_id)
    restored_context = CommandContext(
        context.screen, restored, context.skill_manager, context.context_manager,
        context.client_holder, agent, tmp_path, manager,
    )
    sync_goal_runtime(restored_context)
    assert isinstance(agent.end_policy, GoalPolicy)
    assert agent.end_policy.goal.objective == "迁移全部模块"

    await registry.dispatch("/goal clear", restored_context)
    assert restored.get_goal() is None
    assert agent.end_policy is None
    assert not manager.is_model_tool_enabled("goal")
    restored.close()


@pytest.mark.asyncio
async def test_agent_loop_restarts_half_done_model_and_observes_usage() -> None:
    """模型做一半就停时，harness 自动续跑直至显式 complete。"""
    class HalfDoneClient:
        def __init__(self):
            self.requests = []

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.requests.append((list(messages), list(tools)))
            if len(self.requests) < 3:
                yield TextDelta("只做完一部分")
            elif len(self.requests) == 3:
                yield ToolCallEvent(ToolCall("goal-1", "goal", {"op": "complete"}))
            else:
                yield TextDelta("全部完成")
            yield UsageEvent(8, 2, 10)

    policy = GoalPolicy(Goal("完成全部模块", max_rounds=5))
    manager = ToolManager()
    manager.register_local(*create_goal_tool(lambda: policy))
    client = HalfDoneClient()

    result = await AgentLoop(client, manager, end_policy=policy).run([Message(role="user", content="开始")])

    assert result.stop_reason == "completed"
    assert len(client.requests) == 4
    assert sum(
        "绝不允许把成功重新定义" in message.content
        for messages, _ in client.requests
        for message in messages
    ) >= 2
    assert policy.goal.status == "complete"
    assert policy.goal.tokens_used == 40


@pytest.mark.asyncio
async def test_budget_close_response_has_no_tools() -> None:
    """预算耗尽后的最后一轮只许收尾，不再暴露工具。"""
    class Client:
        def __init__(self):
            self.tools = []

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.tools.append(list(tools))
            yield TextDelta("阶段回复")
            yield UsageEvent(8, 2, 10)

    policy = GoalPolicy(Goal("大任务", max_rounds=0))
    manager = ToolManager()
    manager.register_local(*create_goal_tool(lambda: policy))
    client = Client()

    await AgentLoop(client, manager, end_policy=policy).run([Message(role="user", content="开始")])

    assert len(client.tools) == 2
    assert client.tools[0]
    assert client.tools[1] == []
    assert policy.goal.status == "budget_limited"
