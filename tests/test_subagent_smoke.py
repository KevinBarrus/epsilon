import json

import pytest

from core.config import Settings
from core.model import TextDelta, ToolCall, ToolCallEvent, UsageEvent
from evaluation.subagent_smoke import (
    REQUIRED_EVIDENCE,
    SUBAGENT_SMOKE_PROMPT,
    ScoutBatchRecord,
    SubagentSmokeResult,
    _scout_parallel_result,
    _write_results,
    main,
    run_subagent_smoke_arm,
)


FINAL_ANSWER = " ".join(REQUIRED_EVIDENCE)


def test_subagent_smoke_prompt_contains_independent_parallel_investigations() -> None:
    """测试冒烟任务明确要求并行调查四个独立模块。"""

    assert "四个相互独立的模块" in SUBAGENT_SMOKE_PROMPT
    assert "可并行委派" in SUBAGENT_SMOKE_PROMPT
    for evidence in REQUIRED_EVIDENCE:
        assert evidence in SUBAGENT_SMOKE_PROMPT


class SmokeClient:
    """按工具可见性区分 off、on 父请求与 Scout 请求。"""

    async def stream_response(self, messages, tools=(), thinking_level=None):
        if any(
            message.role == "system" and "你是 Scout" in message.content
            for message in messages
        ):
            yield TextDelta("Scout evidence")
            yield UsageEvent(40, 10, 50)
            return
        has_spawn = any(
            tool["function"]["name"] == "spawn_agent"  # type: ignore[index]
            for tool in tools
        )
        if has_spawn and not any(message.role == "tool" for message in messages):
            for index in range(4):
                yield ToolCallEvent(
                    ToolCall(
                        f"spawn-{index}",
                        "spawn_agent",
                        {"task": f"调查模块 {index}"},
                    )
                )
            yield UsageEvent(80, 20, 100)
            return
        yield TextDelta(FINAL_ANSWER)
        yield UsageEvent(80, 20, 100)


@pytest.mark.asyncio
async def test_subagent_smoke_reports_off_and_on_metrics(tmp_path) -> None:
    """测试冒烟分别报告父 token、全部 token、Scout 次数与摘要字符。"""

    settings = Settings(
        "https://example.com",
        "test-model",
        "key",
        context_window=10_000,
        reserve_tokens=1_000,
        keep_recent_tokens=2_000,
    )

    off = await run_subagent_smoke_arm(
        tmp_path,
        SmokeClient(),
        settings,
        enabled=False,
    )
    on = await run_subagent_smoke_arm(
        tmp_path,
        SmokeClient(),
        settings,
        enabled=True,
    )

    assert off.task_completed is True
    assert off.parent_actual_tokens == 100
    assert off.scout_actual_tokens == 0
    assert off.all_agent_actual_tokens == 100
    assert off.scout_calls == 0
    assert off.parent_scout_result_chars == 0
    assert on.task_completed is True
    assert on.parent_actual_tokens == 200
    assert on.scout_actual_tokens == 200
    assert on.scout_requests_missing_usage == 0
    assert on.scout_missing_usage_reasons == {}
    assert on.all_agent_actual_tokens == 400
    assert on.scout_calls == 4
    assert on.scout_outcomes == {"completed": 4}
    assert on.parent_scout_result_chars == len("Scout evidence") * 4
    assert on.scout_batches == (ScoutBatchRecord("parallel", 4),)
    assert on.scout_parallel is True
    assert on.scout_parallel_summary == "是（同一批次 4 个）"


@pytest.mark.asyncio
async def test_subagent_smoke_keeps_partial_usage_when_requests_are_missing(
    tmp_path,
) -> None:
    """测试 Scout 请求缺 usage 时仍记录已收到部分和缺失请求数。"""

    class MissingScoutUsageClient(SmokeClient):
        async def stream_response(self, messages, tools=(), thinking_level=None):
            if any(
                message.role == "system" and "你是 Scout" in message.content
                for message in messages
            ):
                yield TextDelta("Scout evidence")
                if messages[-1].content.endswith("0"):
                    yield UsageEvent(40, 10, 50)
                return
            async for event in super().stream_response(messages, tools, thinking_level):
                yield event

    result = await run_subagent_smoke_arm(
        tmp_path,
        MissingScoutUsageClient(),
        Settings(
            "https://example.com",
            "test-model",
            "key",
            context_window=10_000,
            reserve_tokens=1_000,
            keep_recent_tokens=2_000,
        ),
        enabled=True,
    )

    assert result.scout_actual_tokens == 50
    assert result.scout_requests_missing_usage == 3
    assert result.scout_missing_usage_reasons == {"completed_missing_usage": 3}
    assert result.all_agent_actual_tokens == result.parent_actual_tokens + 50


def test_scout_parallel_result_reports_split_batches() -> None:
    """测试 Scout 分散在多个批次时明确判为串行委派。"""

    parallel, summary = _scout_parallel_result(
        (
            ScoutBatchRecord("parallel", 2),
            ScoutBatchRecord("parallel", 2),
        )
    )

    assert parallel is False
    assert summary == "否（分散 2 批次）"


def test_subagent_smoke_writes_jsonl(tmp_path) -> None:
    """测试冒烟结果按档位写成可追溯 JSONL。"""

    path = tmp_path / "smoke.jsonl"
    _write_results(
        path,
        [
            SubagentSmokeResult(
                arm="off",
                subagent_enabled=False,
                task_completed=True,
                parent_actual_tokens=100,
                scout_actual_tokens=0,
                scout_requests_missing_usage=0,
                scout_missing_usage_reasons={},
                all_agent_actual_tokens=100,
                duration_ms=10.0,
                scout_calls=0,
                scout_outcomes={},
                scout_batches=(),
                scout_parallel=False,
                scout_parallel_summary="否（未调用 Scout）",
                parent_scout_result_chars=0,
                parent_model_requests=1,
                final_content=FINAL_ANSWER,
            )
        ],
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["arm"] == "off"
    assert record["all_agent_actual_tokens"] == 100


def test_subagent_smoke_requires_explicit_confirmation(
    monkeypatch,
    capsys,
) -> None:
    """测试未确认时不会发起真实模型请求。"""

    monkeypatch.setattr("sys.argv", ["subagent_smoke"])

    assert main() == 2
    assert "--confirm" in capsys.readouterr().out
