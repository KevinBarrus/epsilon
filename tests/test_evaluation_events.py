import pytest

from core.agent_loop import RetryEvent, ToolBatchEvent, ToolExecutionEvent
from core.model import TextDelta, ToolCall, ToolCallEvent, ToolResult, UsageEvent
from core.loop_guard import LoopGuardEvent, action_digest, args_preview, output_fingerprint
from evaluation.events import child_event_record, event_to_record, message_to_record


def test_event_to_record_normalizes_model_and_tool_events() -> None:
    """测试模型和工具事件可以转换为 JSON-safe 记录"""

    assert event_to_record(TextDelta("文本"))["type"] == "assistant_delta"
    assert event_to_record(
        ToolCallEvent(ToolCall("call-1", "read_file", {"path": "a.txt"}))
    )["type"] == "tool_call"
    assert event_to_record(
        ToolExecutionEvent(
            ToolCall("call-1", "read_file", {"path": "a.txt"}),
            ToolResult("call-1", "内容"),
        )
    )["type"] == "tool_result"


def test_event_to_record_normalizes_usage_event() -> None:
    """测试服务端用量事件可以转换为 JSON-safe 记录"""

    assert event_to_record(UsageEvent(12, 3, 15)) == {
        "type": "usage",
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }


def test_event_to_record_normalizes_model_retry() -> None:
    """测试模型重试事件可保存到评测轨迹。"""

    assert event_to_record(RetryEvent(1, 2, 0.5)) == {
        "type": "model_retry",
        "attempt": 1,
        "max_attempts": 2,
        "delay_seconds": 0.5,
    }


def test_event_to_record_normalizes_tool_batch() -> None:
    """测试工具批次轨迹保留模式、数量、耗时与 Worker 标识。"""

    assert event_to_record(
        ToolBatchEvent(
            (ToolCall("call-1", "read_file", {"path": "a.txt"}),),
            "sequential",
            12.5,
        )
    ) == {
        "type": "tool_batch",
        "agent_role": "parent",
        "agent_run_id": "",
        "execution_mode": "sequential",
        "tool_calls": 1,
        "duration_ms": 12.5,
    }


def test_child_event_record_keeps_replay_fields() -> None:
    """子 Agent 记录要带轮次、签名摘要、输出摘要与 Worker 标识，才可离线回放。"""

    call = ToolCall("call-9", "read_file", {"path": "b.txt"})
    batch = child_event_record(
        ToolBatchEvent((call,), "sequential", 1.0, "worker", "spawn-1"),
        round_number=7,
    )
    assert batch is not None
    assert batch["type"] == "batch"
    assert batch["agent_run_id"] == "spawn-1"
    assert batch["round"] == 7
    assert batch["calls"] == [
        {
            "call_id": "call-9",
            "tool": "read_file",
            "args_digest": action_digest("read_file", {"path": "b.txt"}),
            "args_preview": args_preview({"path": "b.txt"}),
        }
    ]

    result = child_event_record(
        ToolExecutionEvent(
            call,
            ToolResult("call-9", "内容", is_error=True, error_category="timeout"),
            "worker",
            "spawn-1",
        )
    )
    assert result is not None
    assert result["agent_run_id"] == "spawn-1"
    assert result["call_id"] == "call-9"
    assert result["args_digest"] == action_digest("read_file", {"path": "b.txt"})
    assert result["error_family"] == "timeout"
    assert result["output_digest"] == output_fingerprint("内容")


def test_child_event_record_keeps_loop_guard_event() -> None:
    """子 Agent 的纠偏事件必须归档，否则回放看不到触发轮次与信号类型。"""

    record = child_event_record(
        LoopGuardEvent("no_progress", "detailed", None, 9, "worker", "spawn-1")
    )
    assert record == {
        "type": "loop_guard",
        "role": "worker",
        "agent_run_id": "spawn-1",
        "kind": "no_progress",
        "level": "detailed",
        "tool_name": None,
        "count": 9,
    }


def test_child_event_record_is_backward_compatible() -> None:
    """旧事件没带 agent_run_id 时不能报错，只落空标识。"""

    record = child_event_record(
        ToolExecutionEvent(
            ToolCall("call-1", "read_file", {"path": "a.txt"}),
            ToolResult("call-1", "内容"),
        )
    )
    assert record is not None and record["agent_run_id"] == ""
    assert child_event_record(object()) is None


def test_message_to_record_keeps_role_and_content() -> None:
    """测试消息轨迹保留角色和内容"""

    record = message_to_record(__import__("core.model", fromlist=["Message"]).Message("user", "你好"))

    assert record == {"type": "user_message", "role": "user", "content": "你好"}
