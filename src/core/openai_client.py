"""OpenAI-compatible 模型客户端实现。"""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from openai import AsyncOpenAI

from .config import Settings
from .errors import ErrorCategory
from .model import (
    Message,
    ModelEvent,
    ModelClientError,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    UsageEvent,
)


@dataclass
class _ToolCallBuffer:
    """暂存流式工具调用的分片内容。"""

    call_id: str = ""
    name: str = ""
    arguments: str = ""


class OpenAICompatibleClient:
    """使用 OpenAI SDK 调用 OpenAI-compatible 服务。"""

    def __init__(
        self,
        settings: Settings,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """根据配置创建客户端，也允许注入测试客户端。"""

        self._model_name = settings.model_name
        self._is_deepseek = _is_deepseek_endpoint(settings.base_url)
        self._first_byte_timeout_seconds = settings.first_byte_timeout_seconds
        self._stream_idle_timeout_seconds = settings.stream_idle_timeout_seconds
        self._stream_usage = settings.stream_usage
        self._client = client or AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=max(
                self._first_byte_timeout_seconds,
                self._stream_idle_timeout_seconds,
            ),
        )

    async def stream_chat(
        self,
        messages: Sequence[Message],
        thinking_level: str | None = None,
    ) -> AsyncIterator[str]:
        """发送消息并逐段返回模型生成的文本。"""

        async for event in self.stream_response(
            messages, thinking_level=thinking_level
        ):
            if isinstance(event, TextDelta):
                yield event.content

    async def stream_response(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, object]] = (),
        thinking_level: str | None = None,
    ) -> AsyncIterator[ModelEvent]:
        """发送消息并解析文本和工具调用事件。"""

        request_messages = [_serialize_message(message) for message in messages]
        request: dict[str, object] = {
            "model": self._model_name,
            "messages": request_messages,
            "stream": True,
        }
        if tools:
            request["tools"] = list(tools)
        _apply_thinking_options(request, thinking_level, self._is_deepseek)
        if self._stream_usage:
            request["stream_options"] = {"include_usage": True}

        try:
            tool_calls: dict[int, _ToolCallBuffer] = {}
            usage: object | None = None
            async for chunk in self._stream_chunks(request):
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield TextDelta(content)
                reasoning = _reasoning_delta(delta)
                if reasoning:
                    yield TextDelta("", reasoning=reasoning)
                for tool_call_delta in getattr(delta, "tool_calls", None) or ():
                    _append_tool_call_delta(tool_calls, tool_call_delta)

            for index in sorted(tool_calls):
                yield ToolCallEvent(_build_tool_call(tool_calls[index], index))
            if usage is not None:
                usage_event = _build_usage_event(usage)
                if usage_event is not None:
                    yield usage_event
        except asyncio.CancelledError:
            raise
        except ModelClientError:
            raise
        except Exception as exc:
            raise _to_model_error(exc) from exc

    async def _stream_chunks(
        self,
        request: Mapping[str, object],
    ) -> AsyncIterator[object]:
        """按首包和分片空闲时间读取模型流。"""

        async with asyncio.timeout(self._first_byte_timeout_seconds):
            stream = await self._client.chat.completions.create(**request)
            iterator = stream.__aiter__()
            try:
                chunk = await anext(iterator)
            except StopAsyncIteration:
                return
        yield chunk

        while True:
            try:
                async with asyncio.timeout(self._stream_idle_timeout_seconds):
                    chunk = await anext(iterator)
            except StopAsyncIteration:
                return
            yield chunk


def _is_deepseek_endpoint(base_url: str) -> bool:
    """判断当前请求是否直连 DeepSeek 服务。"""

    hostname = urlparse(base_url).hostname or ""
    return hostname == "deepseek.com" or hostname.endswith(".deepseek.com")


def _apply_thinking_options(
    request: dict[str, object],
    thinking_level: str | None,
    is_deepseek: bool,
) -> None:
    """按服务端协议写入推理开关与强度参数。"""

    if is_deepseek and thinking_level == "off":
        request["thinking"] = {"type": "disabled"}
        return
    if thinking_level and thinking_level != "off":
        if is_deepseek:
            request["thinking"] = {"type": "enabled"}
        request["reasoning_effort"] = thinking_level


def _reasoning_delta(delta: object) -> str | None:
    """读取 OpenAI-compatible 服务常用的推理分片字段。"""

    for field in ("reasoning_content", "reasoning", "reasoning_text"):
        value = getattr(delta, field, None)
        if isinstance(value, str) and value:
            return value
    return None


def _to_model_error(error: BaseException) -> ModelClientError:
    """将底层模型异常转换为统一类别和安全提示。"""

    error_name = type(error).__name__
    if isinstance(error, (ConnectionError, TimeoutError, asyncio.TimeoutError)) or error_name in {
        "APIConnectionError",
        "APITimeoutError",
    }:
        is_timeout = isinstance(error, (TimeoutError, asyncio.TimeoutError)) or error_name == "APITimeoutError"
        category: ErrorCategory = "timeout" if is_timeout else "network"
        message = "model request timed out" if category == "timeout" else "model network request failed"
        return ModelClientError(message, category=category, retryable=True, cause=error)
    if error_name == "RateLimitError":
        return ModelClientError("model request rate limited", category="rate_limit", retryable=True, cause=error)
    if error_name in {"AuthenticationError", "PermissionDeniedError"}:
        return ModelClientError("model authentication failed, check the API key", category="authentication", cause=error)
    if error_name in {"BadRequestError", "UnprocessableEntityError"}:
        if _is_context_overflow_error(error):
            return ModelClientError(
                "model context limit exceeded, retrying after compaction",
                category="context_overflow",
                cause=error,
            )
        return ModelClientError("model request parameters invalid", category="invalid_request", cause=error)
    return ModelClientError("model request failed, check config and network", category="internal", cause=error)


def _is_context_overflow_error(error: BaseException) -> bool:
    """根据服务端结构化错误码识别上下文长度超限。"""

    return any(
        code in {
            "context_length_exceeded",
            "context_window_exceeded",
            "max_context_length_exceeded",
        }
        for code in _error_codes(error)
    )


def _error_codes(error: BaseException) -> tuple[str, ...]:
    """读取 SDK 异常中可用于分类的结构化错误码。"""

    values = [getattr(error, "code", None)]
    body = getattr(error, "body", None)
    if isinstance(body, Mapping):
        details = body.get("error")
        if isinstance(details, Mapping):
            values.extend((details.get("code"), details.get("type")))
    return tuple(
        value.strip().lower()
        for value in values
        if isinstance(value, str) and value.strip()
    )


def _serialize_message(message: Message) -> dict[str, object]:
    """将内部消息转换为 OpenAI-compatible 消息。"""

    request: dict[str, object] = {
        "role": message.role,
        "content": message.content,
    }
    if message.tool_calls:
        request["tool_calls"] = [
            {
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(
                        tool_call.arguments,
                        ensure_ascii=False,
                    ),
                },
            }
            for tool_call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        request["tool_call_id"] = message.tool_call_id
    return request


def _append_tool_call_delta(
    buffers: dict[int, _ToolCallBuffer],
    delta: object,
) -> None:
    """将一个 OpenAI 工具调用分片追加到对应缓冲区。"""

    index = getattr(delta, "index", 0)
    buffer = buffers.setdefault(index, _ToolCallBuffer())
    call_id = getattr(delta, "id", None)
    if call_id:
        buffer.call_id = call_id
    function = getattr(delta, "function", None)
    if function is None:
        return
    name = getattr(function, "name", None)
    if name:
        buffer.name = name
    arguments = getattr(function, "arguments", None)
    if arguments:
        buffer.arguments += arguments


def _build_tool_call(buffer: _ToolCallBuffer, index: int) -> ToolCall:
    """将工具调用缓冲区转换为已校验的内部对象。"""

    if not buffer.call_id or not buffer.name:
        raise ModelClientError(f"model returned an incomplete tool call: {index}")
    try:
        arguments = json.loads(buffer.arguments or "{}")
    except json.JSONDecodeError as exc:
        raise ModelClientError("model returned tool arguments that are not valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ModelClientError("model tool arguments must be a JSON object")
    return ToolCall(
        call_id=buffer.call_id,
        name=buffer.name,
        arguments=arguments,
    )


def _build_usage_event(usage: object) -> UsageEvent | None:
    """将服务端 usage 分片转换为用量事件，字段不完整时视为缺失。"""

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        total_tokens = prompt_tokens + completion_tokens
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int) or not isinstance(total_tokens, int):
        return None
    # 缓存命中 token 来自 prompt_tokens_details.cached_tokens，缺失时置 None
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", None)
    cached = cached_tokens if isinstance(cached_tokens, int) else None
    return UsageEvent(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached,
    )
