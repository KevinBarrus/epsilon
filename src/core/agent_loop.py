"""实现模型和工具之间的最小执行循环。"""

import asyncio
import random
from asyncio import sleep as yield_to_event_loop
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

from .context import ContextBuildResult, model_request_fingerprint
from .end_policy import EndPolicySummary, TurnEndPolicy
from .error_policy import AgentErrorPolicy
from .errors import AgentError
from .model import (
    Message,
    ModelClient,
    ModelEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    UsageEvent,
)
from .tools import ToolManager


@dataclass(frozen=True)
class ToolExecutionEvent:
    """表示一次工具调用已经完成。"""

    tool_call: ToolCall
    result: ToolResult


@dataclass(frozen=True)
class ToolBatchEvent:
    """表示一批工具调用的执行模式与总耗时。"""

    tool_calls: tuple[ToolCall, ...]
    execution_mode: Literal["parallel", "sequential"]
    duration_ms: float


@dataclass(frozen=True)
class RetryEvent:
    """表示一次模型请求失败后即将重试。"""

    attempt: int
    max_attempts: int
    delay_seconds: float


AgentEvent = ModelEvent | ToolExecutionEvent | ToolBatchEvent | RetryEvent
EventHandler = Callable[[AgentEvent], Awaitable[None]]
ContextBuilder = Callable[[Sequence[Message], bool], Awaitable[ContextBuildResult]]
RETRY_BASE_DELAY_SECONDS = 0.5
RETRY_MAX_DELAY_SECONDS = 4.0
TOOL_CANCELLED_UNKNOWN = "tool call cancelled; execution outcome unknown"


@dataclass(frozen=True)
class AgentRunResult:
    """保存一轮 Agent Loop 的完整运行结果。"""

    messages: tuple[Message, ...]
    final_content: str
    new_messages: tuple[Message, ...] = ()
    stop_reason: Literal["completed", "tool_limit"] = "completed"
    tool_rounds: int = 0
    verification_reminder_injected: bool = False
    write_count: int = 0
    post_write_command_results: tuple[ToolResult, ...] = ()
    verification_command_results: tuple[ToolResult, ...] = ()


class AgentLoopCancelled(asyncio.CancelledError):
    """保存取消前已产生消息的 Agent Loop 取消异常。"""

    def __init__(self, new_messages: tuple[Message, ...]) -> None:
        super().__init__("Agent loop cancelled")
        self.new_messages = new_messages


class AgentLoopFailed(AgentError):
    """保存失败前已完成消息的 Agent Loop 异常。"""

    def __init__(self, error: AgentError, new_messages: tuple[Message, ...]) -> None:
        """保留原始错误信息和本轮已经完成的消息。"""

        super().__init__(
            category=error.category,
            operation=error.operation,
            user_message=error.user_message,
            model_message=error.model_message,
            retryable=error.retryable,
            cause=error.cause,
        )
        self.new_messages = new_messages


class _ToolBatchCancelled(asyncio.CancelledError):
    """携带工具批次取消前已经确定的全部结果。"""

    def __init__(
        self,
        results: tuple[ToolResult, ...],
        unknown_call_ids: frozenset[str],
    ) -> None:
        """保存按原调用顺序排列的结果和状态未知的调用标识。"""

        super().__init__("tool batch cancelled")
        self.results = results
        self.unknown_call_ids = unknown_call_ids


class AgentLoop:
    """负责请求模型、执行工具并把结果继续交给模型。"""

    def __init__(
        self,
        client: ModelClient,
        tool_manager: ToolManager,
        max_tool_rounds: int | None = None,
        thinking_level: str = "high",
        end_policy: TurnEndPolicy | None = None,
    ) -> None:
        """创建 Agent Loop，可选地限制单轮工具调用次数。"""

        if max_tool_rounds is not None and max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be > 0")
        self._client = client
        self._tool_manager = tool_manager
        self._max_tool_rounds = max_tool_rounds
        self._error_policy = AgentErrorPolicy()
        self._thinking_level = thinking_level
        self._show_thinking = True
        self._end_policy = end_policy

    @property
    def thinking_level(self) -> str:
        """当前推理强度档位。"""

        return self._thinking_level

    def set_thinking_level(self, level: str) -> None:
        """切换推理强度档位，后续请求生效。"""

        self._thinking_level = level

    @property
    def show_thinking(self) -> bool:
        """是否展示模型思考过程。"""

        return self._show_thinking

    def set_show_thinking(self, show: bool) -> None:
        """切换思考过程展示，后续流式输出生效。"""

        self._show_thinking = show

    def swap_client(self, client: ModelClient) -> None:
        """热切换模型客户端，供 /model 命令在空闲间隙调用。"""

        self._client = client

    async def run(
        self,
        messages: Sequence[Message],
        on_event: EventHandler | None = None,
        build_context: ContextBuilder | None = None,
    ) -> AgentRunResult:
        """执行一轮模型—工具循环并返回完整上下文。"""

        context = list(messages)
        new_messages: list[Message] = []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_rounds = 0
        try:
            while (
                self._max_tool_rounds is None
                or tool_rounds < self._max_tool_rounds
            ):
                text_parts = []
                tool_calls = []
                latest_usage: UsageEvent | None = None
                request_messages = context
                for force_compaction in (False, True):
                    if build_context is not None:
                        context_result = await build_context(
                            context,
                            force_compaction,
                        )
                        request_messages = context_result.messages
                    try:
                        async for event in self._stream_model_events(
                            request_messages,
                            tools=self._tool_manager.model_tools(),
                            thinking_level=self._thinking_level,
                            on_event=on_event,
                        ):
                            if isinstance(event, TextDelta):
                                text_parts.append(event.content)
                            elif isinstance(event, ToolCallEvent):
                                tool_calls.append(event.tool_call)
                            elif isinstance(event, UsageEvent):
                                latest_usage = event
                            if on_event is not None:
                                await on_event(event)
                            if isinstance(event, TextDelta):
                                # 连续缓冲分片也要让出事件循环，避免界面刷新被饿死
                                await yield_to_event_loop(0)
                        break
                    except AgentError as exc:
                        if exc.category != "context_overflow" or force_compaction:
                            raise

                assistant_content = "".join(text_parts)
                completed_tool_calls = tuple(tool_calls)
                assistant_message = Message(
                    role="assistant",
                    content=assistant_content,
                    tool_calls=completed_tool_calls,
                    usage=latest_usage,
                    request_fingerprint=(
                        model_request_fingerprint(
                            request_messages,
                            self._tool_manager.model_tools(),
                        )
                        if latest_usage is not None
                        else None
                    ),
                )
                context.append(assistant_message)
                new_messages.append(assistant_message)
                text_parts = []
                tool_calls = []
                if not completed_tool_calls:
                    follow_up = (
                        self._end_policy.follow_up_message()
                        if self._end_policy is not None
                        else None
                    )
                    if follow_up is None:
                        return self._run_result(
                            context,
                            assistant_content,
                            new_messages,
                            tool_rounds,
                        )
                    context.append(follow_up)
                    continue

                tool_rounds += 1
                try:
                    results, execution_mode, batch_duration_ms = (
                        await self._execute_tool_batch(
                            completed_tool_calls,
                            on_event,
                        )
                    )
                except _ToolBatchCancelled as exc:
                    for tool_call, result in zip(completed_tool_calls, exc.results):
                        new_messages.append(
                            _tool_result_message(
                                tool_call,
                                result,
                                cancelled=(
                                    tool_call.call_id in exc.unknown_call_ids
                                ),
                            )
                        )
                    raise
                for tool_call, result in zip(completed_tool_calls, results):
                    tool_message = _tool_result_message(tool_call, result)
                    context.append(tool_message)
                    new_messages.append(tool_message)
                if self._end_policy is not None:
                    self._end_policy.observe_tool_results(completed_tool_calls, results)
                if on_event is not None:
                    await on_event(
                        ToolBatchEvent(
                            completed_tool_calls,
                            execution_mode,
                            batch_duration_ms,
                        )
                    )

            return self._run_result(
                context,
                assistant_content,
                new_messages,
                stop_reason="tool_limit",
                tool_rounds=tool_rounds,
            )
        except asyncio.CancelledError as exc:
            if text_parts or tool_calls:
                new_messages.append(
                    Message(
                        role="assistant",
                        content="".join(text_parts),
                        tool_calls=tuple(tool_calls),
                        status="cancelled",
                    )
                )
            else:
                _mark_last_assistant_cancelled(new_messages)
            raise AgentLoopCancelled(tuple(new_messages)) from exc
        except AgentError as exc:
            if not new_messages:
                raise
            raise AgentLoopFailed(exc, tuple(new_messages)) from exc

    def _run_result(
        self,
        context: Sequence[Message],
        final_content: str,
        new_messages: Sequence[Message],
        tool_rounds: int,
        stop_reason: Literal["completed", "tool_limit"] = "completed",
    ) -> AgentRunResult:
        """将可选收尾策略的统计统一写入运行结果。"""

        summary = (
            self._end_policy.summary
            if self._end_policy is not None
            else EndPolicySummary(False, 0, (), ())
        )
        return AgentRunResult(
            tuple(context),
            final_content,
            tuple(new_messages),
            stop_reason,
            tool_rounds,
            summary.verification_reminder_injected,
            summary.write_count,
            summary.post_write_command_results,
            summary.verification_command_results,
        )

    async def _execute_tool_batch(
        self,
        tool_calls: tuple[ToolCall, ...],
        on_event: EventHandler | None,
    ) -> tuple[list[ToolResult], Literal["parallel", "sequential"], float]:
        """顺序预检工具，再按本批执行模式运行并保持结果源顺序。"""

        results: list[ToolResult | None] = [None] * len(tool_calls)
        try:
            prepared_calls = []
            for index, tool_call in enumerate(tool_calls):
                prepared = await self._tool_manager.prepare(tool_call)
                if isinstance(prepared, ToolResult):
                    results[index] = prepared
                    if on_event is not None:
                        await on_event(ToolExecutionEvent(tool_call, prepared))
                else:
                    prepared_calls.append((index, prepared))

            execution_mode: Literal["parallel", "sequential"] = (
                "parallel"
                if prepared_calls
                and all(
                    prepared.definition.execution_mode == "parallel"
                    for _, prepared in prepared_calls
                )
                else "sequential"
            )
            batch_started_at = perf_counter()
            if execution_mode == "parallel":
                await self._execute_parallel_prepared_calls(
                    prepared_calls,
                    results,
                    on_event,
                )
            else:
                for index, prepared in prepared_calls:
                    result = await self._tool_manager.execute_prepared(prepared)
                    results[index] = result
                    if on_event is not None:
                        await on_event(ToolExecutionEvent(prepared.tool_call, result))
            batch_duration_ms = (perf_counter() - batch_started_at) * 1000
            assert all(result is not None for result in results)
            return [result for result in results if result is not None], execution_mode, batch_duration_ms
        except asyncio.CancelledError as exc:
            unknown_call_ids = frozenset(
                tool_call.call_id
                for index, tool_call in enumerate(tool_calls)
                if results[index] is None
            )
            completed_results = tuple(
                result
                if result is not None
                else ToolResult(
                    call_id=tool_calls[index].call_id,
                    content=TOOL_CANCELLED_UNKNOWN,
                    is_error=True,
                    error_category="tool_execution",
                )
                for index, result in enumerate(results)
            )
            raise _ToolBatchCancelled(completed_results, unknown_call_ids) from exc

    async def _execute_parallel_prepared_calls(
        self,
        prepared_calls,
        results: list[ToolResult | None],
        on_event: EventHandler | None,
    ) -> None:
        """并发执行已预检只读工具，完成即通知界面。"""

        async def execute_one(index, prepared):
            return index, prepared.tool_call, await self._tool_manager.execute_prepared(prepared)

        tasks = [
            asyncio.create_task(execute_one(index, prepared))
            for index, prepared in prepared_calls
        ]
        try:
            for completed in asyncio.as_completed(tasks):
                index, tool_call, result = await completed
                results[index] = result
                if on_event is not None:
                    await on_event(ToolExecutionEvent(tool_call, result))
        except BaseException:
            for task in tasks:
                task.cancel()
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            for completed in completed_results:
                if isinstance(completed, tuple):
                    index, _, result = completed
                    results[index] = result
            raise

    async def _stream_model_events(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
        thinking_level: str | None = None,
        on_event: EventHandler | None = None,
    ) -> AsyncIterator[ModelEvent]:
        """在不重复展示部分输出的前提下重试模型请求。"""

        attempt = 0
        while True:
            received_event = False
            try:
                async for event in self._client.stream_response(
                    messages, tools=tools, thinking_level=thinking_level
                ):
                    received_event = True
                    yield event
                return
            except AgentError as error:
                decision = self._error_policy.decide(error)
                if (
                    decision.action != "retry"
                    or received_event
                    or attempt >= decision.max_attempts
                ):
                    raise
                attempt += 1
                delay = _retry_delay_seconds(decision, attempt)
                if on_event is not None:
                    await on_event(
                        RetryEvent(attempt, decision.max_attempts, delay)
                    )
                await asyncio.sleep(delay)


def _mark_last_assistant_cancelled(messages: list[Message]) -> None:
    """将最后一条模型消息标记为取消状态。"""

    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "assistant":
            messages[index] = replace(messages[index], status="cancelled")
            return


def _tool_result_message(
    tool_call: ToolCall,
    result: ToolResult,
    *,
    cancelled: bool = False,
) -> Message:
    """将工具结果转换为保留执行状态的消息。"""

    status = "cancelled" if cancelled else "completed"
    if result.is_error and not cancelled:
        status = "error"
    return Message(
        role="tool",
        content=result.content,
        tool_call_id=tool_call.call_id,
        status=status,
        error_category=result.error_category,
    )


def _retry_delay_seconds(decision, attempt: int) -> float:
    """优先使用服务端等待时间，否则计算带抖动的指数退避。"""

    if decision.delay_seconds:
        return decision.delay_seconds
    delay = min(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), RETRY_MAX_DELAY_SECONDS)
    return delay + random.uniform(0, delay * 0.1)
