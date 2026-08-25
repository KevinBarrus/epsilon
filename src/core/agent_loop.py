"""实现模型和工具之间的最小执行循环。"""

import asyncio
import random
from asyncio import sleep as yield_to_event_loop
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from .context import ContextBuildResult
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
)
from .tools import ToolManager


@dataclass(frozen=True)
class ToolExecutionEvent:
    """表示一次工具调用已经完成。"""

    tool_call: ToolCall
    result: ToolResult


@dataclass(frozen=True)
class RetryEvent:
    """表示一次模型请求失败后即将重试。"""

    attempt: int
    max_attempts: int
    delay_seconds: float


AgentEvent = ModelEvent | ToolExecutionEvent | RetryEvent
EventHandler = Callable[[AgentEvent], Awaitable[None]]
ContextBuilder = Callable[[Sequence[Message], bool], Awaitable[ContextBuildResult]]
RETRY_BASE_DELAY_SECONDS = 0.5
RETRY_MAX_DELAY_SECONDS = 4.0


@dataclass(frozen=True)
class AgentRunResult:
    """保存一轮 Agent Loop 的完整运行结果。"""

    messages: tuple[Message, ...]
    final_content: str
    new_messages: tuple[Message, ...] = ()
    stop_reason: Literal["completed", "tool_limit"] = "completed"
    tool_rounds: int = 0


class AgentLoopCancelled(asyncio.CancelledError):
    """保存取消前已产生消息的 Agent Loop 取消异常。"""

    def __init__(self, new_messages: tuple[Message, ...]) -> None:
        super().__init__("Agent loop cancelled")
        self.new_messages = new_messages


class AgentLoop:
    """负责请求模型、执行工具并把结果继续交给模型。"""

    def __init__(
        self,
        client: ModelClient,
        tool_manager: ToolManager,
        max_tool_rounds: int | None = None,
        thinking_level: str = "high",
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
                )
                context.append(assistant_message)
                new_messages.append(assistant_message)
                text_parts = []
                tool_calls = []
                if not completed_tool_calls:
                    return AgentRunResult(
                        tuple(context),
                        assistant_content,
                        tuple(new_messages),
                        tool_rounds=tool_rounds,
                    )

                tool_rounds += 1
                for tool_call in completed_tool_calls:
                    result = await self._tool_manager.execute(tool_call)
                    tool_message = Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=tool_call.call_id,
                    )
                    context.append(tool_message)
                    new_messages.append(tool_message)
                    if on_event is not None:
                        await on_event(ToolExecutionEvent(tool_call, result))

            return AgentRunResult(
                tuple(context),
                assistant_content,
                tuple(new_messages),
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


def _retry_delay_seconds(decision, attempt: int) -> float:
    """优先使用服务端等待时间，否则计算带抖动的指数退避。"""

    if decision.delay_seconds:
        return decision.delay_seconds
    delay = min(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), RETRY_MAX_DELAY_SECONDS)
    return delay + random.uniform(0, delay * 0.1)
