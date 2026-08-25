import pytest

from core.agent_loop import RetryEvent, ToolBatchEvent, ToolExecutionEvent
from core.model import TextDelta, ToolCall, ToolCallEvent, ToolResult, UsageEvent
from evaluation.events import event_to_record, message_to_record


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
    """测试工具批次轨迹保留模式、数量和耗时。"""

    assert event_to_record(
        ToolBatchEvent(
            (ToolCall("call-1", "read_file", {"path": "a.txt"}),),
            "sequential",
            12.5,
        )
    ) == {
        "type": "tool_batch",
        "execution_mode": "sequential",
        "tool_calls": 1,
        "duration_ms": 12.5,
    }


def test_message_to_record_keeps_role_and_content() -> None:
    """测试消息轨迹保留角色和内容"""

    record = message_to_record(__import__("core.model", fromlist=["Message"]).Message("user", "你好"))

    assert record == {"type": "user_message", "role": "user", "content": "你好"}
