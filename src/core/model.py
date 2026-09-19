"""定义模型客户端与应用之间的最小接口。"""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from .config import Settings
from .errors import AgentError, ErrorCategory


MessageRole = Literal["system", "user", "assistant", "tool"]
MessageStatus = Literal["completed", "cancelled", "error"]


@dataclass(frozen=True)
class ToolCall:
    """模型请求执行的一次工具调用。"""

    call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ToolResult:
    """工具执行后返回给模型的结果。"""

    call_id: str
    content: str
    is_error: bool = False
    error_category: ErrorCategory | None = None


@dataclass(frozen=True)
class TextDelta:
    """模型流式返回的一段文本，content 为正文，reasoning 为思考过程。"""

    content: str
    reasoning: str = ""


@dataclass(frozen=True)
class ToolCallEvent:
    """模型流式响应中完成的一次工具调用。"""

    tool_call: ToolCall


@dataclass(frozen=True)
class UsageEvent:
    """模型服务端返回的一次请求的实际 Token 用量。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int | None = None
    cache_miss_tokens: int | None = None


ModelEvent = TextDelta | ToolCallEvent | UsageEvent


@dataclass(frozen=True)
class Message:
    """一次模型对话中的消息。"""

    role: MessageRole
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    status: MessageStatus = "completed"
    error_category: ErrorCategory | None = None
    usage: UsageEvent | None = None
    request_fingerprint: str | None = None
    reasoning: str = ""


class ModelClientError(AgentError):
    """模型客户端向上层报告的统一模型异常。"""

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = "internal",
        retryable: bool = False,
        cause: BaseException | None = None,
    ) -> None:
        """将旧模型异常接口转换为统一 AgentError。"""

        super().__init__(
            category=category,
            operation="model_request",
            user_message=message,
            model_message=message,
            retryable=retryable,
            cause=cause,
        )


class ModelClient(Protocol):
    """所有模型客户端都需要实现的流式对话接口。"""

    async def stream_chat(
        self,
        messages: Sequence[Message],
    ) -> AsyncIterator[str]:
        """根据消息列表生成文本片段。"""

    async def stream_response(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, object]] = (),
    ) -> AsyncIterator[ModelEvent]:
        """根据消息列表生成文本和工具调用事件。"""


class ClientHolder:
    """可变保存当前模型配置与客户端，供 /model 热切换时统一替换。"""

    def __init__(self, settings: Settings, client: ModelClient) -> None:
        """保存初始配置与客户端。"""

        self.settings = settings
        self.client = client

    def swap(self, settings: Settings, client: ModelClient) -> None:
        """替换为新的配置与客户端。"""

        self.settings = settings
        self.client = client
