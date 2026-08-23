import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from core.config import Settings
from core.errors import AgentError
from core.model import (
    Message,
    ModelClientError,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    UsageEvent,
)
from core.openai_client import OpenAICompatibleClient


class FakeCompletions:
    def __init__(self, chunks: list[object]) -> None:
        """保存假的模型响应片段。"""

        self.chunks = chunks
        self.received: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> AsyncIterator[object]:
        """记录请求参数，并返回假的异步响应流。"""

        self.received = kwargs
        return FakeStream(self.chunks)


class FakeStream:
    def __init__(self, chunks: list[object]) -> None:
        """初始化异步响应流。"""

        self._chunks = iter(chunks)

    def __aiter__(self) -> "FakeStream":
        """返回异步迭代器本身。"""

        return self

    async def __anext__(self) -> object:
        """逐个返回预先准备好的响应片段。"""

        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FakeClient:
    def __init__(self, chunks: list[object]) -> None:
        """组装与 OpenAI SDK 结构相似的测试客户端。"""

        self.completions = FakeCompletions(chunks)
        self.chat = SimpleNamespace(completions=self.completions)


def _settings() -> Settings:
    """构造测试用模型配置。"""

    return Settings(
        base_url="https://example.com/v1",
        model_name="test-model",
        api_key="test-key",
    )


def _deepseek_settings() -> Settings:
    """构造直连 DeepSeek 的测试配置。"""

    return Settings(
        base_url="https://api.deepseek.com/",
        model_name="deepseek-test",
        api_key="test-key",
    )


async def _collect(client: OpenAICompatibleClient, thinking_level: str | None = None) -> str:
    """收集客户端产生的全部文本片段。"""

    result = ""
    async for chunk in client.stream_chat(
        [Message(role="user", content="你好")],
        thinking_level=thinking_level,
    ):
        result += chunk
    return result


@pytest.mark.asyncio
async def test_client_sends_openai_compatible_request_and_streams_response() -> None:
    """测试客户端发送正确请求并返回流式文本。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="你"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="好"))]
            ),
        ]
    )
    client = OpenAICompatibleClient(_settings(), fake_sdk)  # type: ignore[arg-type]

    answer = await _collect(client)

    assert answer == "你好"
    assert fake_sdk.completions.received == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": True,
    }


@pytest.mark.asyncio
async def test_client_sends_reasoning_effort_for_thinking_level() -> None:
    """测试传入推理强度时请求携带 reasoning_effort。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="好"))]
            ),
        ]
    )
    client = OpenAICompatibleClient(_settings(), fake_sdk)  # type: ignore[arg-type]

    await _collect(client, thinking_level="high")

    assert fake_sdk.completions.received["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_client_omits_reasoning_effort_for_off() -> None:
    """测试 off 档位不携带 reasoning_effort。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="好"))]
            ),
        ]
    )
    client = OpenAICompatibleClient(_settings(), fake_sdk)  # type: ignore[arg-type]

    await _collect(client, thinking_level="off")

    assert "reasoning_effort" not in fake_sdk.completions.received


@pytest.mark.asyncio
async def test_client_uses_deepseek_thinking_protocol() -> None:
    """测试直连 DeepSeek 时同时启用推理与传递强度。"""

    fake_sdk = FakeClient(
        [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="好"))])]
    )
    client = OpenAICompatibleClient(_deepseek_settings(), fake_sdk)  # type: ignore[arg-type]

    await _collect(client, thinking_level="high")

    assert fake_sdk.completions.received["extra_body"] == {
        "thinking": {"type": "enabled"}
    }
    assert fake_sdk.completions.received["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_client_disables_deepseek_thinking_for_off() -> None:
    """测试直连 DeepSeek 时 off 显式关闭推理。"""

    fake_sdk = FakeClient(
        [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="好"))])]
    )
    client = OpenAICompatibleClient(_deepseek_settings(), fake_sdk)  # type: ignore[arg-type]

    await _collect(client, thinking_level="off")

    assert fake_sdk.completions.received["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert "reasoning_effort" not in fake_sdk.completions.received


@pytest.mark.asyncio
async def test_client_skips_empty_stream_chunks() -> None:
    """测试客户端会跳过没有文本内容的响应片段。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(choices=[]),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="完成"))]
            ),
        ]
    )
    client = OpenAICompatibleClient(_settings(), fake_sdk)  # type: ignore[arg-type]

    assert await _collect(client) == "完成"


@pytest.mark.asyncio
async def test_client_wraps_request_error() -> None:
    """测试底层请求异常会被转换为统一的模型异常。"""

    class FailingCompletions:
        async def create(self, **kwargs: object) -> AsyncIterator[object]:
            """模拟底层网络请求失败。"""

            raise ConnectionError("test failure")

    failing_sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )
    client = OpenAICompatibleClient(_settings(), failing_sdk)  # type: ignore[arg-type]

    with pytest.raises(ModelClientError, match="model network request failed") as error_info:
        await _collect(client)

    error = error_info.value
    assert isinstance(error, AgentError)
    assert error.category == "network"
    assert error.retryable


@pytest.mark.asyncio
async def test_client_classifies_timeout_error() -> None:
    """测试超时异常会被转换为可重试的统一错误。"""

    class TimeoutCompletions:
        async def create(self, **kwargs: object) -> AsyncIterator[object]:
            """模拟模型请求超时。"""

            raise TimeoutError("test timeout")

    timeout_sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=TimeoutCompletions())
    )
    client = OpenAICompatibleClient(_settings(), timeout_sdk)  # type: ignore[arg-type]

    with pytest.raises(ModelClientError) as error_info:
        await _collect(client)

    assert error_info.value.category == "timeout"
    assert error_info.value.retryable


@pytest.mark.asyncio
async def test_client_times_out_when_stream_hangs() -> None:
    """测试流式请求挂起时会在配置时间后返回超时错误。"""

    class SlowStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Event().wait()

    class SlowCompletions:
        async def create(self, **kwargs: object) -> AsyncIterator[object]:
            return SlowStream()

    settings = Settings(
        base_url="https://example.com/v1",
        model_name="test-model",
        api_key="test-key",
        request_timeout_seconds=0.01,
    )
    slow_sdk = SimpleNamespace(chat=SimpleNamespace(completions=SlowCompletions()))
    client = OpenAICompatibleClient(settings, slow_sdk)  # type: ignore[arg-type]

    with pytest.raises(ModelClientError, match="model request timed out") as error_info:
        await _collect(client)

    assert error_info.value.category == "timeout"


@pytest.mark.asyncio
async def test_client_allows_long_stream_with_active_chunks() -> None:
    """测试持续输出的长流不会被累计时长中断。"""

    class DelayedStream:
        def __init__(self) -> None:
            self._chunks = iter(("一", "二", "三", "四"))

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(0.02)
            try:
                content = next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
            )

    class DelayedCompletions:
        async def create(self, **kwargs: object) -> AsyncIterator[object]:
            return DelayedStream()

    settings = Settings(
        base_url="https://example.com/v1",
        model_name="test-model",
        api_key="test-key",
        first_byte_timeout_seconds=0.06,
        stream_idle_timeout_seconds=0.06,
    )
    delayed_sdk = SimpleNamespace(chat=SimpleNamespace(completions=DelayedCompletions()))
    client = OpenAICompatibleClient(settings, delayed_sdk)  # type: ignore[arg-type]

    assert await _collect(client) == "一二三四"


@pytest.mark.asyncio
async def test_client_times_out_when_stream_becomes_idle() -> None:
    """测试两个流式分片之间超时会停止请求。"""

    class IdleStream:
        def __init__(self) -> None:
            self._calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self._calls += 1
            await asyncio.sleep(0.01 if self._calls == 1 else 0.08)
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="片段"))]
            )

    class IdleCompletions:
        async def create(self, **kwargs: object) -> AsyncIterator[object]:
            return IdleStream()

    settings = Settings(
        base_url="https://example.com/v1",
        model_name="test-model",
        api_key="test-key",
        first_byte_timeout_seconds=0.04,
        stream_idle_timeout_seconds=0.04,
    )
    idle_sdk = SimpleNamespace(chat=SimpleNamespace(completions=IdleCompletions()))
    client = OpenAICompatibleClient(settings, idle_sdk)  # type: ignore[arg-type]

    with pytest.raises(ModelClientError, match="model request timed out") as error_info:
        await _collect(client)

    assert error_info.value.category == "timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attributes",
    (
        {"code": "context_length_exceeded"},
        {"body": {"error": {"type": "context_window_exceeded"}}},
    ),
)
async def test_client_classifies_context_overflow_by_structured_code(
    attributes: dict[str, object],
) -> None:
    """测试服务端结构化上下文错误码会被单独分类。"""

    class BadRequestError(Exception):
        def __init__(self, message: str, **kwargs: object) -> None:
            super().__init__(message)
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FailingCompletions:
        async def create(self, **kwargs: object) -> AsyncIterator[object]:
            raise BadRequestError("context limit exceeded", **attributes)

    failing_sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )
    client = OpenAICompatibleClient(_settings(), failing_sdk)  # type: ignore[arg-type]

    with pytest.raises(ModelClientError) as error_info:
        await _collect(client)

    assert error_info.value.category == "context_overflow"


@pytest.mark.asyncio
async def test_client_does_not_guess_context_overflow_from_error_text() -> None:
    """测试没有结构化错误码时不会按错误文本猜测上下文超限。"""

    class BadRequestError(Exception):
        pass

    class FailingCompletions:
        async def create(self, **kwargs: object) -> AsyncIterator[object]:
            raise BadRequestError("maximum context length exceeded")

    failing_sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )
    client = OpenAICompatibleClient(_settings(), failing_sdk)  # type: ignore[arg-type]

    with pytest.raises(ModelClientError) as error_info:
        await _collect(client)

    assert error_info.value.category == "invalid_request"


@pytest.mark.asyncio
async def test_client_classifies_openai_connection_error() -> None:
    """测试 OpenAI SDK 网络异常会被转换为 network 类别"""

    class APIConnectionError(Exception):
        pass

    class FailingCompletions:
        async def create(self, **kwargs: object) -> AsyncIterator[object]:
            """模拟 OpenAI SDK 网络连接失败"""

            raise APIConnectionError("connection failed")

    failing_sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )
    client = OpenAICompatibleClient(_settings(), failing_sdk)  # type: ignore[arg-type]

    with pytest.raises(ModelClientError) as error_info:
        await _collect(client)

    assert error_info.value.category == "network"
    assert error_info.value.retryable


async def _collect_events(client: OpenAICompatibleClient) -> list[object]:
    """收集客户端产生的模型事件。"""

    events: list[object] = []
    async for event in client.stream_response(
        [Message(role="user", content="读取文件")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取文件",
                    "parameters": {"type": "object"},
                },
            }
        ],
    ):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_client_parses_text_and_streaming_tool_call() -> None:
    """测试客户端解析文本片段和分片工具调用。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="开始"))]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="read_file",
                                        arguments='{"path":',
                                    ),
                                )
                            ],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    function=SimpleNamespace(
                                        name=None,
                                        arguments='"README.md"}',
                                    ),
                                )
                            ],
                        )
                    )
                ]
            ),
        ]
    )
    client = OpenAICompatibleClient(_settings(), fake_sdk)  # type: ignore[arg-type]

    events = await _collect_events(client)

    assert events == [
        TextDelta("开始"),
        ToolCallEvent(
            tool_call=ToolCall(
                call_id="call-1",
                name="read_file",
                arguments={"path": "README.md"},
            )
        ),
    ]
    assert fake_sdk.completions.received is not None
    assert fake_sdk.completions.received["tools"]


@pytest.mark.asyncio
async def test_client_rejects_invalid_tool_arguments() -> None:
    """测试客户端拒绝无效的工具参数 JSON。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="read_file",
                                        arguments="{invalid",
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )
        ]
    )
    client = OpenAICompatibleClient(_settings(), fake_sdk)  # type: ignore[arg-type]

    with pytest.raises(ModelClientError, match="not valid JSON"):
        await _collect_events(client)


@pytest.mark.asyncio
async def test_client_omits_stream_options_when_stream_usage_disabled() -> None:
    """测试默认关闭 usage 采集时不发送 stream_options。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))]
            )
        ]
    )
    client = OpenAICompatibleClient(_settings(), fake_sdk)  # type: ignore[arg-type]

    await _collect_events(client)

    assert fake_sdk.completions.received is not None
    assert "stream_options" not in fake_sdk.completions.received


@pytest.mark.asyncio
async def test_client_emits_usage_event_when_stream_usage_enabled() -> None:
    """测试开启 usage 采集时请求携带 stream_options 并产出用量事件。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="完成"))]
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=3,
                    total_tokens=15,
                ),
            ),
        ]
    )
    settings = Settings(
        base_url="https://example.com/v1",
        model_name="test-model",
        api_key="test-key",
        stream_usage=True,
    )
    client = OpenAICompatibleClient(settings, fake_sdk)  # type: ignore[arg-type]

    events = await _collect_events(client)

    assert events == [
        TextDelta("完成"),
        UsageEvent(prompt_tokens=12, completion_tokens=3, total_tokens=15),
    ]
    assert fake_sdk.completions.received is not None
    assert fake_sdk.completions.received["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_client_collects_cached_tokens_from_details() -> None:
    """测试缓存命中 token 从 prompt_tokens_details.cached_tokens 采集。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=1000,
                    completion_tokens=10,
                    total_tokens=1010,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=900),
                ),
            ),
        ]
    )
    settings = Settings(
        base_url="https://example.com/v1",
        model_name="test-model",
        api_key="test-key",
        stream_usage=True,
    )
    client = OpenAICompatibleClient(settings, fake_sdk)  # type: ignore[arg-type]

    events = await _collect_events(client)

    assert events == [
        UsageEvent(
            prompt_tokens=1000,
            completion_tokens=10,
            total_tokens=1010,
            cached_tokens=900,
        )
    ]


@pytest.mark.asyncio
async def test_client_omits_cached_tokens_when_details_missing() -> None:
    """测试服务端未返回缓存明细时 cached_tokens 为 None。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=3,
                    total_tokens=15,
                ),
            ),
        ]
    )
    settings = Settings(
        base_url="https://example.com/v1",
        model_name="test-model",
        api_key="test-key",
        stream_usage=True,
    )
    client = OpenAICompatibleClient(settings, fake_sdk)  # type: ignore[arg-type]

    events = await _collect_events(client)

    assert events == [
        UsageEvent(prompt_tokens=12, completion_tokens=3, total_tokens=15)
    ]


@pytest.mark.asyncio
async def test_client_skips_incomplete_usage() -> None:
    """测试服务端 usage 字段不完整时不产出用量事件。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))]
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                ),
            ),
        ]
    )
    settings = Settings(
        base_url="https://example.com/v1",
        model_name="test-model",
        api_key="test-key",
        stream_usage=True,
    )
    client = OpenAICompatibleClient(settings, fake_sdk)  # type: ignore[arg-type]

    assert await _collect_events(client) == [TextDelta("ok")]


@pytest.mark.asyncio
async def test_client_streams_reasoning_content() -> None:
    """测试客户端采集 DeepSeek 的 reasoning_content 作为思考事件。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content="分析"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="结论"))]
            ),
        ]
    )
    client = OpenAICompatibleClient(_settings(), fake_sdk)  # type: ignore[arg-type]

    events = [event async for event in client.stream_response([Message(role="user", content="hi")])]

    reasoning = [event.reasoning for event in events if getattr(event, "reasoning", "")]
    content = [event.content for event in events if getattr(event, "content", "")]

    assert reasoning == ["分析"]
    assert content == ["结论"]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["reasoning", "reasoning_text"])
async def test_client_streams_openai_compatible_reasoning_fields(field: str) -> None:
    """测试客户端兼容非标准推理字段。"""

    fake_sdk = FakeClient(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(**{field: "分析"}))]
            )
        ]
    )
    client = OpenAICompatibleClient(_settings(), fake_sdk)  # type: ignore[arg-type]

    events = [event async for event in client.stream_response([Message(role="user", content="hi")])]

    assert events == [TextDelta("", reasoning="分析")]
