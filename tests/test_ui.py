import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

import core.ui as ui
from core.context import CONTEXT_FALLBACK_NOTICE
from core.errors import AgentError
from core.model import Message
from core.model import ModelClientError
from core.screen import ChatScreen
from core.session import Session
from core.session_store import SessionStore
from core.status import create_status_info
from core.config import McpStdioSettings, Settings


class FakeClient:
    """记录提交的消息，并返回固定的模型回复"""

    def __init__(self) -> None:
        """初始化请求记录"""

        self.received: list[list[Message]] = []

    async def stream_chat(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        """记录当前请求并返回一段测试文本"""

        self.received.append(list(messages))
        yield "测试回复"


@pytest.mark.asyncio
async def test_submit_handler_sends_conversation_history(tmp_path: Path) -> None:
    """测试连续请求会携带当前会话的完整历史"""

    client = FakeClient()
    session = Session(tmp_path)
    status = create_status_info("test-model", "暂不可查询", tmp_path)
    screen = ChatScreen(status)

    async def handle_submit(prompt: str) -> None:
        """模拟应用层同步记忆和界面的请求流程"""

        screen.add_entry("user", prompt)
        response_index = screen.add_entry("assistant", "")
        session.add_user_message(prompt)
        response = ""
        async for chunk in client.stream_chat(session.get_messages()):
            response += chunk
            screen.append_to_entry(response_index, chunk)
        session.add_assistant_message(response)

    screen._on_submit = handle_submit
    await screen._submit("你好")
    await screen._submit("第二次输入")

    assert client.received == [
        [Message(role="user", content="你好")],
        [
            Message(role="user", content="你好"),
            Message(role="assistant", content="测试回复"),
            Message(role="user", content="第二次输入"),
        ],
    ]


class CancellingClient:
    """返回部分文本后取消请求的测试客户端"""

    def __init__(self) -> None:
        """初始化请求记录"""

        self.received: list[list[Message]] = []

    async def stream_chat(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        """记录请求并在返回部分文本后触发取消"""

        self.received.append(list(messages))
        yield "部分回复"
        raise asyncio.CancelledError


class FailingClient:
    """请求时返回模型错误的测试客户端"""

    async def stream_chat(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        """模拟模型请求失败"""

        raise ModelClientError("请求失败")
        yield ""


@pytest.mark.asyncio
async def test_run_chat_retries_once_with_forced_context_compaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试服务端上下文超限后会强制压缩并重试一次。"""

    class ContextOverflowClient:
        def __init__(self) -> None:
            self.requests: list[list[Message]] = []

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                raise AgentError(
                    category="context_overflow",
                    operation="model_request",
                    user_message="模型上下文超出限制",
                )
            yield ui.TextDelta("已压缩后重试")

    class FakeScreen:
        def __init__(self, status, on_submit, command_names=None, model_name_provider=None, balance_text_provider=None, provider_name_provider=None, thinking_level_provider=None, info_line_provider=None, copy_hint_provider=None, on_copy=None, startup_info_provider=None) -> None:
            self._on_submit = on_submit
            self.application = self
            self.entries: list[tuple[str, str]] = []

        def add_entry(self, role: str, content: str, style: str = "") -> int:
            self.entries.append((role, content))
            return len(self.entries) - 1

        def add_active_entry(self, role: str, content: str, style: str = "") -> int:
            return self.add_entry(role, content, style)

        def commit_entry(self, index: int) -> bool:
            return True

        def add_history_entries(self, entries) -> None:
            self.entries.extend(entries)

        def append_to_entry(self, index: int, content: str) -> None:
            role, current = self.entries[index]
            self.entries[index] = (role, current + content)

        def set_entry_content(self, index: int, content: str) -> None:
            role, _ = self.entries[index]
            self.entries[index] = (role, content)

        def set_tool_result(self, index: int, content: str) -> None:
            return None
        def set_status_message(self, message: str) -> None:
            pass


        def set_working(self, message: str | None, show_elapsed: bool = True) -> None:
            return None

        async def request_approval(self, definition, tool_call, allow_session=True):
            raise AssertionError("本测试不应请求工具审批")

        async def run_async(self) -> None:
            await self._on_submit("继续完成任务")

    client = ContextOverflowClient()
    monkeypatch.setattr(ui, "ChatScreen", FakeScreen)

    await ui.run_chat(
        client,
        create_status_info("test-model", "暂不可查询", tmp_path),
        workspace=tmp_path,
        session_id="00000000-0000-0000-0000-000000000001",
        settings=Settings("https://example.com", "test", "key"),
    )

    assert len(client.requests) == 2
    assert client.requests[0][0] == Message(
        role="system",
        content=ui.AGENT_SYSTEM_PROMPT.replace("{model_name}", "test"),
    )
    assert any(
        message.content == CONTEXT_FALLBACK_NOTICE
        for message in client.requests[1]
    )
    assert all(
        message.role != "system"
        for message in SessionStore(tmp_path).load_messages(
            "00000000-0000-0000-0000-000000000001"
        )
    )


@pytest.mark.asyncio
async def test_cancelled_response_is_kept_in_memory(tmp_path: Path) -> None:
    """测试取消请求后已生成的部分回复会进入下一轮历史"""

    client = CancellingClient()
    session = Session(tmp_path)
    status = create_status_info("test-model", "暂不可查询", tmp_path)
    screen = ChatScreen(status)

    async def handle_submit(prompt: str) -> None:
        """模拟带取消处理的应用层请求流程"""

        screen.add_entry("user", prompt)
        response_index = screen.add_entry("assistant", "")
        session.add_user_message(prompt)
        response = ""
        try:
            async for chunk in client.stream_chat(session.get_messages()):
                response += chunk
                screen.append_to_entry(response_index, chunk)
        except asyncio.CancelledError:
            if response:
                session.add_message(
                    Message(
                        role="assistant",
                        content=response,
                        status="cancelled",
                    )
                )
            screen.append_to_entry(response_index, "(cancelled)")
            raise

    screen._on_submit = handle_submit
    with pytest.raises(asyncio.CancelledError):
        await handle_submit("第一次输入")

    assert session.flush_persistence()
    restored = Session.restore(tmp_path, session.session_id)
    assert restored.get_messages() == [
        Message(role="user", content="第一次输入"),
        Message(
            role="assistant",
            content="部分回复",
            status="cancelled",
        ),
    ]

    # 第二次请求仍然取消，但断言的是它收到的历史
    with pytest.raises(asyncio.CancelledError):
        await handle_submit("第二次输入")

    assert client.received[1] == [
        Message(role="user", content="第一次输入"),
        Message(
            role="assistant",
            content="部分回复",
            status="cancelled",
        ),
        Message(role="user", content="第二次输入"),
    ]


@pytest.mark.asyncio
async def test_model_error_is_saved_with_error_status(tmp_path: Path) -> None:
    """测试模型错误会以异常状态写入会话历史"""

    client = FailingClient()
    session = Session(tmp_path)
    status = create_status_info("test-model", "暂不可查询", tmp_path)
    screen = ChatScreen(status)

    async def handle_submit(prompt: str) -> None:
        """模拟模型错误处理流程"""

        screen.add_entry("user", prompt)
        response_index = screen.add_entry("assistant", "")
        session.add_user_message(prompt)
        try:
            async for chunk in client.stream_chat(session.get_messages()):
                screen.append_to_entry(response_index, chunk)
        except ModelClientError as exc:
            session.add_message(
                Message(
                    role="assistant",
                    content="",
                    status="error",
                    error_category=exc.category,
                )
            )
            screen.append_to_entry(response_index, f"Error: {exc}")

    screen._on_submit = handle_submit
    await screen._submit("测试错误")

    assert session.get_messages() == [
        Message(role="user", content="测试错误"),
        Message(
            role="assistant",
            content="",
            status="error",
            error_category="internal",
        ),
    ]


@pytest.mark.asyncio
async def test_run_chat_refreshes_balance_after_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试每轮对话后刷新余额并更新状态栏显示。"""

    class BalanceProbe:
        """记录查询次数并返回固定余额。"""

        def __init__(self) -> None:
            self.calls = 0

        async def get_balance(self) -> str:
            self.calls += 1
            return "9.99 CNY"

    class FakeScreen:
        """记录构造参数并在提交后读取余额 provider。"""

        instances: list["FakeScreen"] = []

        def __init__(
            self,
            status,
            on_submit,
            command_names=None,
            model_name_provider=None,
            balance_text_provider=None,
            provider_name_provider=None,
            thinking_level_provider=None,
            info_line_provider=None,
            copy_hint_provider=None,
            on_copy=None,
            startup_info_provider=None,
        ) -> None:
            self._on_submit = on_submit
            self.application = self
            self.entries: list[tuple[str, str]] = []
            self.balance_text_provider = balance_text_provider
            FakeScreen.instances.append(self)

        def add_entry(self, role: str, content: str, style: str = "") -> int:
            self.entries.append((role, content))
            return len(self.entries) - 1

        def add_active_entry(self, role: str, content: str, style: str = "") -> int:
            return self.add_entry(role, content, style)

        def commit_entry(self, index: int) -> bool:
            return True

        def add_history_entries(self, entries) -> None:
            pass

        def append_to_entry(self, index: int, content: str) -> None:
            pass

        def set_entry_content(self, index: int, content: str) -> None:
            pass

        def set_tool_result(self, index: int, content: str) -> None:
            return None
        def set_status_message(self, message: str) -> None:
            pass


        def set_working(self, message: str | None, show_elapsed: bool = True) -> None:
            return None

        def invalidate(self) -> None:
            pass

        async def request_approval(self, definition, tool_call, allow_session=True):
            raise AssertionError("不应请求工具审批")

        async def request_skill_picker(self, items, checked):
            raise AssertionError("不应请求 skill 选择器")

        async def request_choice_picker(self, items, title, extra_options=None):
            raise AssertionError("不应请求选择器")

        async def request_text_input(self, title, is_password=False):
            raise AssertionError("不应请求文本输入")

        async def run_async(self) -> None:
            await self._on_submit("继续")

    class FakeTurnClient:
        """实现 stream_response 的测试客户端。"""

        async def stream_chat(self, messages):
            yield "测试回复"

        async def stream_response(self, messages, tools=(), thinking_level=None):
            yield ui.TextDelta("测试回复")

    client = FakeTurnClient()
    probe = BalanceProbe()
    FakeScreen.instances = []
    monkeypatch.setattr(ui, "ChatScreen", FakeScreen)

    await ui.run_chat(
        client,
        create_status_info("test-model", "暂不可查询", tmp_path),
        Settings("https://example.com", "test", "key"),
        workspace=tmp_path,
        balance_provider=probe,
    )

    assert probe.calls >= 1
    assert FakeScreen.instances[0].balance_text_provider() == "9.99 CNY"


class ThinkingClient:
    """先返回思考过程再返回正文的测试客户端。"""

    async def stream_response(self, messages, tools=(), thinking_level=None):
        yield ui.TextDelta("", reasoning="推理中")
        yield ui.TextDelta("结论文本")


class _ThinkingScreen:
    """记录思考与回复条目的假界面。"""

    instances: list["_ThinkingScreen"] = []

    def __init__(self, status, on_submit, command_names=None, **kwargs) -> None:
        self._on_submit = on_submit
        self.application = self
        self.entries: list[tuple[str, str]] = []
        _ThinkingScreen.instances.append(self)

    def add_entry(self, role: str, content: str, style: str = "") -> int:
        self.entries.append((role, content))
        return len(self.entries) - 1

    def add_active_entry(self, role: str, content: str, style: str = "") -> int:
        return self.add_entry(role, content, style)

    def commit_entry(self, index: int) -> bool:
        return True

    def add_history_entries(self, entries) -> None:
        self.entries.extend(entries)

    def append_to_entry(self, index: int, content: str) -> None:
        role, current = self.entries[index]
        self.entries[index] = (role, current + content)

    def set_entry_content(self, index: int, content: str) -> None:
        role, _ = self.entries[index]
        self.entries[index] = (role, content)

    def set_tool_result(self, index: int, content: str) -> None:
        return None
    def set_entry_style(self, index: int, style: str) -> None:
        return None

    def set_working(self, message: str | None, show_elapsed: bool = True) -> None:
        return None

    async def request_approval(self, definition, tool_call, allow_session=True):
        return None

    async def run_async(self) -> None:
        await self._on_submit("继续完成任务")


@pytest.mark.asyncio
async def test_reasoning_renders_as_thinking_entry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试思考过程合并进回复条目（\x00 标记包裹）。"""

    from core.status import create_status_info
    from core.config import Settings

    _ThinkingScreen.instances = []
    monkeypatch.setattr(ui, "ChatScreen", _ThinkingScreen)

    await ui.run_chat(
        ThinkingClient(),
        create_status_info("test-model", "暂不可查询", tmp_path),
        Settings("https://example.com", "test", "key"),
        workspace=tmp_path,
    )

    screen = _ThinkingScreen.instances[0]
    roles = [role for role, _ in screen.entries]

    assert roles[0] == "user"
    assert roles.count("assistant") == 1
    assert "thinking" not in roles
    reply = next(
        content for role, content in screen.entries if role == "assistant"
    )
    assert reply == "\x00推理中\x00结论文本"


@pytest.mark.asyncio
async def test_reasoning_hidden_shows_thinking_label(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试关闭思考展示时只显示 Thinking... 标签。"""

    from core.status import create_status_info
    from core.config import Settings
    from core.agent_loop import AgentLoop
    from core.tools.manager import ToolManager

    _ThinkingScreen.instances = []
    monkeypatch.setattr(ui, "ChatScreen", _ThinkingScreen)
    hidden_loop = AgentLoop(
        ThinkingClient(),
        ToolManager(),
        thinking_level="high",
    )
    hidden_loop.set_show_thinking(False)

    await ui.run_chat(
        ThinkingClient(),
        create_status_info("test-model", "暂不可查询", tmp_path),
        Settings("https://example.com", "test", "key"),
        workspace=tmp_path,
        agent_loop=hidden_loop,
    )

    screen = _ThinkingScreen.instances[0]
    reply = next(
        content for role, content in screen.entries if role == "assistant"
    )

    assert "Thinking..." in reply
    assert "推理中" not in reply
