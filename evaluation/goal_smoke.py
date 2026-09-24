"""用真实模型验证：阶段性停顿会被 GoalPolicy 自动续跑。"""

import argparse
import asyncio
import json
from pathlib import Path

from core.agent_loop import AgentLoop
from core.config import load_settings
from core.goal import Goal, GoalPolicy, create_goal_tool
from core.model import Message, TextDelta
from core.openai_client import OpenAICompatibleClient
from core.tools import ToolManager

from .online import TimedModelClient


async def run_smoke() -> dict[str, object]:
    """让模型首轮只完成第一阶段，再检查续跑与显式完成。"""
    settings = load_settings()
    client = TimedModelClient(OpenAICompatibleClient(settings))
    goal = Goal("分两阶段输出甲、乙两个词，且第二阶段后显式声明目标完成", max_rounds=3, token_budget=20_000)
    visible_text: list[str] = []
    policy = GoalPolicy(goal, completion_check=lambda: "乙" in "".join(visible_text))
    tools = ToolManager()
    tools.register_local(*create_goal_tool(lambda: policy))
    prompt = (
        "这是目标机制的两阶段冒烟：第一次回复只写‘阶段一：甲’，不要写乙，也不要调用工具。"
        "收到系统的续跑消息后，再写‘阶段二：乙’，随后调用 goal(op=complete)。"
        "第一阶段结束时直接停止回复；不要把第一阶段当作目标完成。"
    )
    async def collect_event(event: object) -> None:
        """只把模型正文计入完成检查，不把任务提示算作证据。"""
        if isinstance(event, TextDelta) and not event.reasoning:
            visible_text.append(event.content)

    try:
        result = await AgentLoop(client, tools, thinking_level="low", end_policy=policy).run(
            [policy.instruction_message(), Message(role="user", content=prompt)],
            on_event=collect_event,
        )
        assistants = [message for message in result.new_messages if message.role == "assistant"]
        first_stopped_halfway = bool(
            assistants and not assistants[0].tool_calls
            and "甲" in assistants[0].content and "乙" not in assistants[0].content
        )
        continuation_sent = any(
            message.role == "system" and "绝不允许把成功重新定义" in message.content
            for message in result.messages
        )
        return {
            "model_name": settings.model_name,
            "passed": first_stopped_halfway and continuation_sent and goal.status == "complete",
            "first_stopped_halfway": first_stopped_halfway,
            "continuation_sent": continuation_sent,
            "goal_status": goal.status,
            "rounds_started": goal.rounds_started,
            "model_requests": len(client.requests),
            "actual_tokens": client.total_actual_tokens,
            "requests_missing_usage": sum(usage is None for usage in client.usages),
            "duration_ms": round(sum(client.durations_ms), 1),
            "assistant_messages": [message.content for message in assistants],
            "stop_reason": result.stop_reason,
        }
    finally:
        await client.close()


def main() -> int:
    """仅在显式确认后发起付费请求并保存完整冒烟记录。"""
    parser = argparse.ArgumentParser(description="Goal 机制真机冒烟")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("evaluation-results/goal-smoke.jsonl"))
    args = parser.parse_args()
    if not args.confirm:
        print("本脚本会发起真实模型请求；请先确认费用，再加 --confirm 运行。")
        return 2
    result = asyncio.run(run_smoke())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
