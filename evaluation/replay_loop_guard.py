"""离线回放 Loop Guard：用归档的 `child_events.jsonl` 重跑空转判定。

归档里的 batch 记录带轮次与调用签名摘要，result 记录带错误族与输出指纹，
因此可以按 `agent_run_id` 分组、逐轮喂给 guard，精确回答：

- 每个子 Agent 跑了几轮、guard 会注入几次、第一次在第几轮；
- 各信号（无进展 / 错误族 / 重复调用 / abab）分别触发多少次。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from core.loop_guard import ActionFact, LoopGuardConfig, ToolCallLoopGuard


def tool_capabilities() -> dict[str, str | None]:
    """按真实工具定义推导 capability，避免与生产实现漂移。"""

    from core.tools import (
        ToolManager,
        create_edit_file_tool,
        create_list_files_tool,
        create_read_file_tool,
        create_run_command_tool,
        create_search_files_tool,
        create_write_file_tool,
    )

    manager = ToolManager()
    workspace = Path(".")
    for create_tool in (
        create_read_file_tool,
        create_list_files_tool,
        create_search_files_tool,
        create_write_file_tool,
        create_edit_file_tool,
        create_run_command_tool,
    ):
        manager.register_local(*create_tool(workspace))
    capabilities: dict[str, str | None] = {
        definition.name: definition.capability
        for definition in manager.list_definitions()
    }
    # 委派工具的 capability 由角色决定，目标文件不在本仓库内，这里按定义补齐
    capabilities.update(
        {
            "spawn_agent": "agent.scout",
            "spawn_worker": "agent.worker",
            "spawn_reviewer": "agent.reviewer",
            "goal": None,
        }
    )
    return capabilities


@dataclass
class RunReplay:
    """一个子 Agent 运行的回放结果。"""

    run_id: str
    role: str
    rounds: int = 0
    injections: int = 0
    first_trigger_round: int | None = None
    longest_no_fact_streak: int = 0
    signals: Counter[str] = field(default_factory=Counter)


def replay(
    path: Path,
    config: LoopGuardConfig | None = None,
) -> dict[str, RunReplay]:
    """按 run_id 分组逐轮重跑 guard，返回每个 run 的回放结果。"""

    runs: dict[str, RunReplay] = {}
    results: dict[tuple[str, str], dict[str, object]] = {}
    batches: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        kind = record.get("type")
        run_id = str(record.get("agent_run_id", ""))
        if kind == "result":
            results[(run_id, str(record.get("call_id", "")))] = record
        elif kind == "batch":
            batches.append(record)
            runs.setdefault(run_id, RunReplay(run_id, str(record.get("role", ""))))

    capabilities = tool_capabilities()
    guards: dict[str, ToolCallLoopGuard] = {}
    for record in batches:
        run_id = str(record.get("agent_run_id", ""))
        run = runs[run_id]
        guard = guards.get(run_id)
        if guard is None:
            guard = ToolCallLoopGuard(config, role=run.role or "parent")
            guards[run_id] = guard
        run.rounds += 1

        facts = []
        for call in record.get("calls", []):  # type: ignore[union-attr]
            tool = str(call.get("tool", ""))
            result = results.get((run_id, str(call.get("call_id", ""))), {})
            is_error = bool(result.get("is_error"))
            facts.append(
                ActionFact(
                    tool=tool,
                    args_digest=str(call.get("args_digest", "")),
                    args_preview=str(call.get("args_preview", "")),
                    is_error=is_error,
                    error_family=result.get("error_family"),  # type: ignore[arg-type]
                    output_fingerprint=(
                        str(result.get("output_digest"))
                        if tool == "run_command" and not is_error
                        else None
                    ),
                )
            )

        injection = guard.observe_facts(facts, capabilities)
        run.longest_no_fact_streak = max(
            run.longest_no_fact_streak, guard.rounds_without_new_fact
        )
        if injection is not None:
            run.injections += 1
            run.signals[injection.event.kind] += 1
            if run.first_trigger_round is None:
                run.first_trigger_round = run.rounds
    return runs


def format_report(runs: dict[str, RunReplay]) -> str:
    """把回放结果渲染成人读的文本报告。"""

    if not runs:
        return "归档里没有 batch 记录（旧格式），无法按轮回放。"
    lines = [
        "run_id                          role      rounds  inject  first  streak  signals",
    ]
    for run in sorted(runs.values(), key=lambda item: item.rounds, reverse=True):
        signals = ",".join(f"{kind}:{count}" for kind, count in run.signals.most_common()) or "-"
        first = "-" if run.first_trigger_round is None else str(run.first_trigger_round)
        lines.append(
            f"{run.run_id:30.30}  {run.role:8.8}  {run.rounds:6}  {run.injections:6}  "
            f"{first:5}  {run.longest_no_fact_streak:6}  {signals}"
        )
    total = sum(run.injections for run in runs.values())
    lines.append(f"合计：{len(runs)} 个子 Agent 运行，guard 会注入 {total} 次。")
    return "\n".join(lines)


def main() -> int:
    """命令行入口：回放一个 child_events.jsonl。"""

    parser = argparse.ArgumentParser(description="离线回放 Loop Guard 判定")
    parser.add_argument("path", type=Path, help="child_events.jsonl 路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument("--thresholds", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--no-progress-rounds", type=int, default=8)
    args = parser.parse_args()

    config = LoopGuardConfig(
        thresholds=tuple(args.thresholds),
        no_progress_rounds=args.no_progress_rounds,
    )
    runs = replay(args.path, config)
    if args.json:
        print(
            json.dumps(
                {
                    run_id: {
                        "role": run.role,
                        "rounds": run.rounds,
                        "injections": run.injections,
                        "first_trigger_round": run.first_trigger_round,
                        "longest_no_fact_streak": run.longest_no_fact_streak,
                        "signals": dict(run.signals),
                    }
                    for run_id, run in runs.items()
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_report(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
