"""将 Agent 运行事件转换为可保存的评测轨迹"""

from core.agent_loop import (
    RetryEvent,
    ToolBatchEvent,
    ToolCallCancelledEvent,
    ToolCallStartedEvent,
    ToolExecutionEvent,
)
from core.goal import CompletionGateEvent
from core.loop_guard import (
    LoopGuardEvent,
    action_digest,
    args_preview,
    error_family,
    output_fingerprint,
)
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
            "agent_role": event.agent_role,
            "agent_run_id": event.agent_run_id,
            "call_id": event.tool_call.call_id,
            "name": event.tool_call.name,
            "content": event.result.content,
            "is_error": event.result.is_error,
            "error_category": event.result.error_category,
        }
    if isinstance(event, ToolCallStartedEvent):
        return {
            "type": "tool_started",
            "agent_role": event.agent_role,
            "agent_run_id": event.agent_run_id,
            "call_id": event.tool_call.call_id,
            "name": event.tool_call.name,
            "arguments": event.tool_call.arguments,
        }
    if isinstance(event, ToolCallCancelledEvent):
        return {
            "type": "tool_cancelled",
            "agent_role": event.agent_role,
            "agent_run_id": event.agent_run_id,
            "call_id": event.tool_call.call_id,
            "name": event.tool_call.name,
            "arguments": event.tool_call.arguments,
        }
    if isinstance(event, ToolBatchEvent):
        return {
            "type": "tool_batch",
            "agent_role": event.agent_role,
            "agent_run_id": event.agent_run_id,
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
    if isinstance(event, LoopGuardEvent):
        return {
            "type": "loop_guard",
            "agent_role": event.agent_role,
            "agent_run_id": event.agent_run_id,
            "kind": event.kind,
            "level": event.level,
            "tool_name": event.tool_name,
            "count": event.count,
        }
    if isinstance(event, CompletionGateEvent):
        return {
            "type": "completion_gate",
            "verdict": event.verdict,
            "unmet": list(event.unmet),
            "attempt": event.attempt,
            "accepted": event.accepted,
            "verified": event.verified,
            "reason": event.reason,
        }
    if isinstance(event, UsageEvent):
        return {
            "type": "usage",
            "prompt_tokens": event.prompt_tokens,
            "completion_tokens": event.completion_tokens,
            "total_tokens": event.total_tokens,
        }
    raise TypeError(f"不支持的评测事件：{type(event).__name__}")


def child_event_record(event: object, round_number: int = 0) -> dict[str, object] | None:
    """把子 Agent 事件转成可离线回放的归档记录，非采集事件返回 None。

    记录里带上轮次、调用签名摘要、输出摘要与 Worker 标识，使轨迹能按 run_id
    分组后精确重跑 Loop Guard 的判定。
    """

    if isinstance(event, ToolBatchEvent):
        return {
            "type": "batch",
            "role": event.agent_role,
            "agent_run_id": event.agent_run_id,
            "round": round_number,
            "execution_mode": event.execution_mode,
            "calls": [
                {
                    "call_id": call.call_id,
                    "tool": call.name,
                    "args_digest": action_digest(call.name, call.arguments),
                    "args_preview": args_preview(call.arguments),
                }
                for call in event.tool_calls
            ],
        }
    if isinstance(event, ToolExecutionEvent):
        return {
            "type": "result",
            "role": event.agent_role,
            "agent_run_id": event.agent_run_id,
            "call_id": event.tool_call.call_id,
            "tool": event.tool_call.name,
            "is_error": event.result.is_error,
            "error_category": event.result.error_category,
            "error_family": error_family(
                event.result.is_error,
                event.result.error_category,
                event.result.content,
            ),
            "args_digest": action_digest(event.tool_call.name, event.tool_call.arguments),
            "args_preview": args_preview(event.tool_call.arguments),
            "output_digest": output_fingerprint(event.result.content),
        }
    if isinstance(event, RetryEvent):
        return {"type": "retry", "attempt": event.attempt}
    if isinstance(event, LoopGuardEvent):
        return {
            "type": "loop_guard",
            "role": event.agent_role,
            "agent_run_id": event.agent_run_id,
            "kind": event.kind,
            "level": event.level,
            "tool_name": event.tool_name,
            "count": event.count,
        }
    return None


def message_to_record(message: Message) -> dict[str, object]:
    """将一条消息转换为评测轨迹记录"""

    return {
        "type": f"{message.role}_message",
        "role": message.role,
        "content": message.content,
    }
