"""将 Agent 运行事件转换为可保存的评测轨迹"""

from core.agent_loop import RetryEvent, ToolBatchEvent, ToolExecutionEvent
from core.model import Message, TextDelta, ToolCallEvent, UsageEvent


def event_to_record(event: object) -> dict[str, object]:
    """将一个 Agent 事件转换为 JSON-safe 记录"""

    if isinstance(event, TextDelta):
        return {"type": "assistant_delta", "content": event.content}
    if isinstance(event, ToolCallEvent):
        return {
            "type": "tool_call",
            "call_id": event.tool_call.call_id,
            "name": event.tool_call.name,
            "arguments": event.tool_call.arguments,
        }
    if isinstance(event, ToolExecutionEvent):
        return {
            "type": "tool_result",
            "call_id": event.tool_call.call_id,
            "name": event.tool_call.name,
            "content": event.result.content,
            "is_error": event.result.is_error,
            "error_category": event.result.error_category,
        }
    if isinstance(event, ToolBatchEvent):
        return {
            "type": "tool_batch",
            "execution_mode": event.execution_mode,
            "tool_calls": len(event.tool_calls),
            "duration_ms": event.duration_ms,
        }
    if isinstance(event, RetryEvent):
        return {
            "type": "model_retry",
            "attempt": event.attempt,
            "max_attempts": event.max_attempts,
            "delay_seconds": event.delay_seconds,
        }
    if isinstance(event, UsageEvent):
        return {
            "type": "usage",
            "prompt_tokens": event.prompt_tokens,
            "completion_tokens": event.completion_tokens,
            "total_tokens": event.total_tokens,
        }
    raise TypeError(f"不支持的评测事件：{type(event).__name__}")


def message_to_record(message: Message) -> dict[str, object]:
    """将一条消息转换为评测轨迹记录"""

    return {
        "type": f"{message.role}_message",
        "role": message.role,
        "content": message.content,
    }
