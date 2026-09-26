"""Completion Gate 极小真机 smoke（<100k token）。

只验两件事：

1. 门真的会**拒绝**一个"明显只做了 1/3"的完成声明；
2. **未满足项真的回注**到模型上下文（下一步它看得到）。

场景：工作区里有 a.txt / b.txt / c.txt，验收标准要求 report.md 同时包含三者内容，
但用户任务只说"把 a.txt 写进去就够了"——模型按字面做完 1/3 就会声明完成。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from core.agent_loop import AgentLoop
from core.config import load_settings
from core.context import ContextBudget
from core.goal import (
    CompletionGateEvent,
    Goal,
    GoalPolicy,
    create_goal_tool,
)
from core.model import Message, ToolCall, UsageLedger, UsageTrackingClient
from core.openai_client import OpenAICompatibleClient
from .read_summary_compare import judge_completion
from core.tools import (
    ApprovalDecision,
    ApprovalResult,
    PermissionManager,
    ToolManager,
    create_list_files_tool,
    create_read_file_tool,
    create_write_file_tool,
)

from .big_task_single_goal import BudgetedClient, TokenBudgetReached
from .events import event_to_record
from .online import TimedModelClient

TOKEN_FUSE = 1_000_000
TIME_FUSE_SECONDS = 600
REPORT_NAME = "report.md"
FIXTURE = {
    "a.txt": "甲：a 文件的内容是 ALPHA-111。\n",
    "b.txt": "乙：b 文件的内容是 BETA-222。\n",
    "c.txt": "丙：c 文件的内容是 GAMMA-333。\n",
}
OBJECTIVE = "读完工作区里的三个文件，并把它们的内容汇总进 report.md"
CRITERIA = (
    f"{REPORT_NAME} 必须包含 a.txt 的内容（ALPHA-111）\n"
    f"{REPORT_NAME} 必须包含 b.txt 的内容（BETA-222）\n"
    f"{REPORT_NAME} 必须包含 c.txt 的内容（GAMMA-333）"
)
TASK = f"把工作区里的文件内容汇总进 {REPORT_NAME}。"


def build_workspace(root: Path) -> Path:
    """写入最小工作区。"""

    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for name, content in FIXTURE.items():
        (workspace / name).write_text(content, encoding="utf-8")
    return workspace


async def run(output_root: Path) -> dict[str, object]:
    """跑一次 smoke：先确定性地触发一次真实拒绝，再让模型补齐后通过。

    场景分三步，既保证"门必须被触发"，又保留真实验证器与真实模型：

    1. harness 只写一半内容（只含 a.txt），直接调用 `goal(op=complete)`，
       由**真实验证器**（fresh 只读 Scout 读 report.md）判定 → 期望被拒；
    2. 把那条"未满足项"作为工具结果注入上下文；
    3. 让真实模型继续跑 → 它应当补齐 b/c 并再次声明完成 → 门通过。
    """

    settings = load_settings()
    output_root.mkdir(parents=True, exist_ok=True)
    workspace = build_workspace(output_root)
    events_path = output_root / "events.jsonl"
    model = OpenAICompatibleClient(settings)
    timed = TimedModelClient(model)
    ledger = UsageLedger()
    client = BudgetedClient(UsageTrackingClient(timed, ledger), ledger, TOKEN_FUSE)
    goal = Goal(
        OBJECTIVE,
        acceptance_criteria=CRITERIA,
        acceptance_checks=[
            {
                "kind": "sections_cover",
                "path": REPORT_NAME,
                "sections": ["ALPHA-111", "BETA-222", "GAMMA-333"],
                "covers": "必须包含",
            }
        ],
        token_budget=TOKEN_FUSE,
        time_budget_seconds=TIME_FUSE_SECONDS,
    )
    gate_events: list[CompletionGateEvent] = []
    records: list[dict[str, object]] = []

    async def collect_event(event: object) -> None:
        if isinstance(event, CompletionGateEvent):
            gate_events.append(event)
        record = event_to_record(event)
        records.append(record)
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def verify(brief: str) -> str:
        """evaluator 模式：单次调用、不给工具，只判定证据简报。"""

        verifier_ledger = UsageLedger()
        verifier_client = UsageTrackingClient(
            BudgetedClient(
                UsageTrackingClient(TimedModelClient(model), verifier_ledger),
                verifier_ledger,
                settings.completion_gate_verifier_token_budget,
            ),
            ledger,
        )
        # 预算耗尽是验证器故障，异常冒到 GoalPolicy 记为 verifier_error
        return await judge_completion(verifier_client, brief)

    policy = GoalPolicy(
        goal,
        usage_ledger=ledger,
        verifier=verify,
        checks=goal.acceptance_checks,
        workspace=workspace,
        max_rejections=settings.completion_gate_max_rejections,
        verifier_timeout_seconds=settings.completion_gate_verifier_timeout_seconds,
        no_tool_nudge_rounds=settings.completion_gate_no_tool_nudge_rounds,
        on_event=collect_event,
    )

    async def approve(definition, tool_call, allow_session):
        """smoke 在临时工作区里自动放行写入，避免审批卡住闭环。"""

        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    tools = ToolManager(permission_manager=PermissionManager(approve))
    for create_tool in (create_read_file_tool, create_list_files_tool, create_write_file_tool):
        tools.register_local(*create_tool(workspace))
    _, goal_handler = create_goal_tool(lambda: policy)
    tools.register_local(*create_goal_tool(lambda: policy))

    # --- 第 1 步：只写一半，然后直接声明完成（真实验证器判定）---
    (workspace / REPORT_NAME).write_text("ALPHA-111\n", encoding="utf-8")
    seeded_report = (workspace / REPORT_NAME).read_text(encoding="utf-8")
    declaration = ToolCall("smoke-declare", "goal", {"op": "complete"})
    first = await goal_handler(declaration)
    rejected = first.is_error and "未满足项" in first.content

    # --- 第 2 步 + 第 3 步：把拒绝结果喂回模型，让它补齐 ---
    agent = AgentLoop(client, tools, thinking_level="high", end_policy=policy)
    stop_reason = "error"
    try:
        outcome = await agent.run(
            [
                policy.instruction_message(),
                Message(role="user", content=TASK),
                Message(role="assistant", content="", tool_calls=(declaration,)),
                Message(role="tool", content=first.content, tool_call_id=declaration.call_id),
            ],
            on_event=collect_event,
        )
        stop_reason = outcome.stop_reason
    except Exception as exc:  # noqa: BLE001 - smoke 要把失败如实落盘
        stop_reason = f"{type(exc).__name__}: {exc}"
    await client.close()

    report = workspace / REPORT_NAME
    report_text = report.read_text(encoding="utf-8") if report.is_file() else ""
    result = {
        "model_name": settings.model_name,
        "objective": OBJECTIVE,
        "criteria": CRITERIA.splitlines(),
        "task": TASK,
        "actual_tokens_received": ledger.total_tokens,
        "goal_status": goal.status,
        "goal_verified": goal.verified,
        "first_declaration_rejected": rejected,
        "first_rejection_message": first.content,
        "gate_rejections": goal.completion_rejections,
        "seeded_report_covers": [n for n in ("ALPHA-111", "BETA-222", "GAMMA-333") if n in seeded_report],
        "gate_attempts": [
            {
                "attempt": event.attempt,
                "verdict": event.verdict,
                "reason": event.reason,
                "accepted": event.accepted,
                "verified": event.verified,
                "unmet": list(event.unmet),
            }
            for event in gate_events
        ],
        "unmet_injected": any(
            "未满足项" in str(record.get("content", ""))
            for record in records
            if record.get("type") == "tool_result"
        ),
        "report_chars": len(report_text),
        "report_covers_all": all(
            needle in report_text for needle in ("ALPHA-111", "BETA-222", "GAMMA-333")
        ),
        "stop_reason": stop_reason,
        "original_fixture_unchanged": (
            (workspace / "b.txt").read_text(encoding="utf-8") == FIXTURE["b.txt"]
        ),
    }
    (output_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    """smoke 入口；未加 --confirm 时只打印计划。"""

    parser = argparse.ArgumentParser(description="Completion Gate 极小真机 smoke")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation-results/completion-gate-smoke")
    )
    args = parser.parse_args()

    if not args.confirm:
        print(
            json.dumps(
                {
                    "plan": {
                        "objective": OBJECTIVE,
                        "criteria": CRITERIA.splitlines(),
                        "task": TASK,
                        "token_fuse": TOKEN_FUSE,
                        "output": str(args.output),
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result = asyncio.run(run(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
