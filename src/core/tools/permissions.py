"""实现工具权限判断和外部确认回调。"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

from ..model import ToolCall
from .types import ToolDefinition


class PermissionDenied(PermissionError):
    """工具没有获得执行权限时抛出的异常。"""


class ApprovalDecision(Enum):
    """表示用户对一次工具调用作出的审批决定。"""

    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"


@dataclass(frozen=True)
class ApprovalResult:
    """封装审批决定和可选的用户反馈。"""

    decision: ApprovalDecision
    feedback: str = ""


ApprovalHandler = Callable[
    [ToolDefinition, ToolCall, bool],
    Awaitable[ApprovalResult],
]


class PermissionManager:
    """集中判断工具是否可以执行，不依赖 TUI 实现。"""

    def __init__(self, approval_handler: ApprovalHandler | None = None) -> None:
        """记录应用层回调和当前 Session 的工具授权。"""

        self._approval_handler = approval_handler
        self._session_grants: set[tuple[str, str]] = set()

    async def authorize(
        self,
        definition: ToolDefinition,
        tool_call: ToolCall,
    ) -> ApprovalResult:
        """自动放行安全工具，其它工具交给外部确认。"""

        if definition.permission == "read":
            return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

        grant_key = (definition.source, definition.name)
        if grant_key in self._session_grants:
            return ApprovalResult(ApprovalDecision.ALLOW_SESSION)

        if self._approval_handler is None:
            raise PermissionDenied("this tool requires user approval")

        allow_session = definition.permission != "command" and definition.idempotent
        result = await self._approval_handler(definition, tool_call, allow_session)
        if result.decision == ApprovalDecision.ALLOW_SESSION and not allow_session:
            return ApprovalResult(ApprovalDecision.ALLOW_ONCE, result.feedback)
        if result.decision == ApprovalDecision.ALLOW_SESSION:
            self._session_grants.add(grant_key)
        return result
