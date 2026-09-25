"""fork vs fresh 子 Agent 的微基准：同一个"父已读过 X"的场景，量 token / 缓存命中 / 读调用。

场景是确定性的：harness 自己让父读了目标文件、构造父上下文快照，然后分别用
`fresh` 与 `fork` 起一个子 Agent，问同一个问题。因此不依赖模型是否愿意委派，
能直接量出"fork 到底省了什么"。

**会调真实模型**（子 Agent 的提问与回答走模型），需要 --confirm。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from core.agent_loop import PARENT_CONTEXT, ToolExecutionEvent
from core.config import load_settings
from core.context import ContextBudget
from core.model import Message, ToolCall
from core.openai_client import OpenAICompatibleClient
from core.subagent import create_spawn_agent_tool
from core.tools import ToolManager, create_read_file_tool

from .online import TimedModelClient

TOKEN_FUSE = 2_000_000
QUESTION = "{path} 里说了什么？"
ANSWER_FACT = "租约 60 秒"

FIXTURE = {
    "docs/architecture.md": (
        "# 架构说明\n\n调度器采用“租约 60 秒”的心跳续租机制；worker 崩溃后由租约过期触发重领。\n"
        "检索链路是向量召回 + BM25 并行，再用 RRF 融合。\n"
    ),
    "docs/notes.md": "# 备注\n\n这里只是无关的备注内容。\n",
    "src/scheduler.py": "LEASE_SECONDS = 60\n",
}


def build_workspace(root: Path) -> Path:
    """写入一个最小的真实感项目，供子 Agent 阅读。"""

    workspace = root / "workspace"
    for relative, content in FIXTURE.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return workspace


async def measure(mode: str, workspace: Path, settings, target: str) -> dict[str, object]:
    """用指定模式跑一次子 Agent，返回 token / 缓存命中 / 读调用次数。"""

    # 父先自己读一遍目标文件，构造真实的父上下文快照
    manager = ToolManager()
    manager.register_local(*create_read_file_tool(workspace))
    read_result = await manager.execute(ToolCall("parent-read", "read_file", {"path": target}))
    snapshot = (
        Message(role="user", content="先熟悉一下这个项目的架构。"),
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall("parent-read", "read_file", {"path": target}),),
        ),
        Message(role="tool", content=read_result.content, tool_call_id="parent-read"),
    )

    client = TimedModelClient(OpenAICompatibleClient(settings))
    metrics: list[object] = []
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    _, handler = create_spawn_agent_tool(
        workspace,
        lambda: client,
        lambda: "high",
        ContextBudget(settings.context_window or 100_000, settings.reserve_tokens, settings.keep_recent_tokens),
        on_metrics=metrics.append,
        on_event=collect,
        force_mode=mode,
    )

    started = perf_counter()
    token = PARENT_CONTEXT.set(snapshot)
    try:
        result = await handler(
            ToolCall(f"spawn-{mode}", "spawn_agent", {"task": QUESTION.format(path=target)})
        )
    finally:
        PARENT_CONTEXT.reset(token)
    await client.close()

    reads = [
        event
        for event in events
        if isinstance(event, ToolExecutionEvent) and event.tool_call.name == "read_file"
    ]
    metric = metrics[0]
    return {
        "mode": mode,
        "tokens": metric.total_tokens,
        "cache_hit_tokens": metric.cache_hit_tokens,
        "prompt_tokens": metric.prompt_tokens,
        "cache_hit_rate": metric.cache_hit_rate,
        "read_calls": len(reads),
        "answered_with_fact": ANSWER_FACT in result.content,
        "duration_seconds": round(perf_counter() - started, 2),
    }


async def run(target: str, output_root: Path) -> dict[str, object]:
    """跑 fresh 与 fork 两档并落盘结果。"""

    settings = load_settings()
    output_root.mkdir(parents=True, exist_ok=True)
    workspace = build_workspace(output_root)
    ledger = UsageLedger()
    results = []
    for mode in ("fresh", "fork", "fork_last_n"):
        results.append(await measure(mode, workspace, settings, target))
    (output_root / "result.json").write_text(
        json.dumps(
            {
                "target": target,
                "model_name": settings.model_name,
                "thinking": "high",
                "token_fuse": TOKEN_FUSE,
                "results": results,
                "total_tokens": sum(item["tokens"] for item in results),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"results": results}


def main() -> int:
    """微基准入口；未加 --confirm 时只做准备与打印。"""

    parser = argparse.ArgumentParser(description="fork vs fresh 子 Agent 微基准")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--target", default="docs/architecture.md")
    parser.add_argument("--output", type=Path, default=Path("evaluation-results/fork-fresh-smoke"))
    args = parser.parse_args()

    if not args.confirm:
        print(
            json.dumps(
                {
                    "plan": {
                        "target": args.target,
                        "arms": ["fresh", "fork", "fork_last_n"],
                        "token_fuse": TOKEN_FUSE,
                        "output": str(args.output),
                    }
                },
                ensure_ascii=False,
            )
        )
        return 0

    result = asyncio.run(run(args.target, args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
