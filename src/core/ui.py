"""实现全屏终端对话界面"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from .agent_loop import (
    AgentLoop,
    AgentLoopCancelled,
    AgentLoopError,
    RetryEvent,
    ToolExecutionEvent,
)
from .errors import AgentError
from .screen import ChatScreen
from .status import StatusInfo
from .agent_loop import ToolExecutionEvent
from .model import Message, ModelClient, TextDelta, ToolCallEvent, UsageEvent
from .commands import (
    CommandContext,
    CommandRegistry,
    start_skill_command,
    stop_skill_command,
    model_command_slash,
    thinking_command_slash,
    skills_command_slash,
    mcp_command_slash,
    compact_command_slash,
    status_command_slash,
    copy_command_slash,
    clear_command_slash,
    quit_command_slash,
    export_command_slash,
    diff_command_slash,
    background_image_command_slash,
    delete_command_slash,
    thinking_toggle_command_slash,
)
from .balance import UNAVAILABLE_BALANCE, BalanceProvider
from .config import Settings
from .cost import UsageTotals, cache_hit_rate, format_tokens
from .setup import infer_provider
from .context import ContextBudget, ContextManager, DEFAULT_CONTEXT_BUDGET
from .prompts import load_prompt
from .session import Session
from .skills import SkillManager
from .model import ClientHolder
from .tools import (
    PermissionManager,
    ToolManager,
    StdioMcpProvider,
    create_edit_file_tool,
    create_list_files_tool,
    create_read_file_tool,
    create_run_command_tool,
    create_search_files_tool,
    create_write_file_tool,
)


AGENT_SYSTEM_PROMPT = load_prompt("agent")


@dataclass(frozen=True)
class ChatExitInfo:
    """描述一次界面结束后可由启动层输出的最小信息"""

    session_id: str | None
    usage_totals: UsageTotals | None


def _default_command_registry() -> CommandRegistry:
    """创建注册了全部内置命令的注册表。"""

    registry = CommandRegistry()
    registry.register(start_skill_command)
    registry.register(stop_skill_command)
    registry.register(model_command_slash)
    registry.register(thinking_command_slash)
    registry.register(skills_command_slash)
    registry.register(mcp_command_slash)
    registry.register(compact_command_slash)
    registry.register(status_command_slash)
    registry.register(copy_command_slash)
    registry.register(clear_command_slash)
    registry.register(quit_command_slash)
    registry.register(export_command_slash)
    registry.register(diff_command_slash)
    registry.register(background_image_command_slash)
    registry.register(delete_command_slash)
    registry.register(thinking_toggle_command_slash)
    return registry


async def run_chat(
    client: ModelClient,
    status: StatusInfo,
    settings: Settings,
    workspace: Path | None = None,
    session_id: str | None = None,
    context_budget: ContextBudget | None = None,
    balance_provider: BalanceProvider | None = None,
    mcp_provider: StdioMcpProvider | None = None,
    max_tool_rounds: int = 10,
    agent_loop: AgentLoop | None = None,
) -> ChatExitInfo:
    """启动全屏界面，并处理模型的流式回复"""

    screen: ChatScreen
    session_workspace = (workspace or Path.cwd()).resolve()
    current_balance = status.balance
    usage_totals = UsageTotals()
    latest_usage: UsageEvent | None = None
    copy_hint = ""

    async def refresh_balance() -> None:
        """每轮对话后刷新余额，失败保留旧值。"""

        if balance_provider is None:
            return
        try:
            refreshed = await balance_provider.get_balance()
        except Exception:
            return
        if refreshed != UNAVAILABLE_BALANCE:
            nonlocal current_balance
            current_balance = refreshed
            screen.application.invalidate()

    session = (
        Session.restore(session_workspace, session_id)
        if session_id
        else Session(session_workspace)
    )
    skill_manager = SkillManager(session_workspace)
    command_registry = _default_command_registry()
    client_holder = ClientHolder(settings, client)

    async def handle_submit(prompt: str) -> None:
        """发送请求，并同步当前会话的消息历史"""

        # slash command 优先于普通用户消息
        if prompt.startswith("/"):
            command_context = CommandContext(
                screen,
                session,
                skill_manager,
                context_manager,
                client_holder,
                agent_loop,
                session_workspace,
                tool_manager,
            )
            if await command_registry.dispatch(prompt, command_context):
                return
            screen.add_entry("tool", f"Unknown command: {prompt}")
            return

        # 先更新界面，让用户立即看到本轮输入和待生成的回复区域
        screen.add_entry("user", prompt)
        response_index = screen.add_active_entry("assistant", "")
        response_parts: list[str] = []
        thinking_open = False
        tool_activity_indices: dict[str, int] = {}
        awaiting_response_after_tool = False
        screen.set_working(
            "thinking",
            show_elapsed=not agent_loop.show_thinking,
        )

        # 用户消息必须先进入会话，模型才能在本轮请求中看到它
        session.add_user_message(prompt)
        new_compactions = []
        fallback_used = False

        async def build_context(messages, force_compaction: bool):
            """按完整运行时历史构建下一次模型请求上下文。"""

            nonlocal fallback_used
            result = await context_manager.build_for_model_result(
                client_holder.client,
                messages,
                [*session.get_compactions(), *new_compactions],
                force_compaction=force_compaction,
            )
            if result.compaction is not None:
                new_compactions.append(result.compaction)
            fallback_used = fallback_used or result.fallback_used
            return result

        async def handle_event(event) -> None:
            """将模型事件转换为回复文本或简短工具活动条目。"""

            nonlocal response_index, thinking_open, awaiting_response_after_tool, latest_usage

            if isinstance(event, TextDelta):
                if event.reasoning:
                    # 思考过程合并进回复条目（\x00 标记包裹，渲染时斜体灰）
                    if awaiting_response_after_tool:
                        response_index = screen.add_active_entry("assistant", "")
                        awaiting_response_after_tool = False
                        response_parts.clear()
                        thinking_open = False
                    if not thinking_open:
                        screen.append_to_entry(response_index, "\x00")
                        thinking_open = True
                        if agent_loop.show_thinking:
                            screen.append_to_entry(response_index, event.reasoning)
                        else:
                            screen.append_to_entry(response_index, "Thinking...")
                    elif agent_loop.show_thinking:
                        screen.append_to_entry(response_index, event.reasoning)
                else:
                    if awaiting_response_after_tool:
                        response_index = screen.add_active_entry("assistant", "")
                        awaiting_response_after_tool = False
                        response_parts.clear()
                    if thinking_open:
                        # 闭合思考块标记，正文使用普通样式
                        screen.append_to_entry(response_index, "\x00")
                        thinking_open = False
                    response_parts.append(event.content)
                    screen.append_to_entry(response_index, event.content)
            elif isinstance(event, ToolCallEvent):
                screen.commit_entry(response_index)
                summary = _tool_call_summary(event.tool_call)
                tool_activity_indices[event.tool_call.call_id] = screen.add_active_entry(
                    "tool",
                    summary,
                    style="class:tool-pending",
                )
                awaiting_response_after_tool = True
                screen.set_working(f"running {event.tool_call.name}")
            elif isinstance(event, RetryEvent):
                screen.set_working(
                    f"Retrying ({event.attempt}/{event.max_attempts}) "
                    f"in {event.delay_seconds:.0f}s"
                )
            elif isinstance(event, ToolExecutionEvent):
                index = tool_activity_indices.get(event.tool_call.call_id)
                if index is not None:
                    if event.result.is_error:
                        screen.set_entry_style(index, "class:tool-error")
                        _update_tool_result(screen, index, event, success=False)
                    else:
                        screen.set_entry_style(index, "class:tool-success")
                        _update_tool_result(screen, index, event)
                    screen.commit_entry(index)
                screen.set_working(
                    "thinking",
                    show_elapsed=not agent_loop.show_thinking,
                )
            elif isinstance(event, UsageEvent):
                latest_usage = event
                usage_totals.add(event, client_holder.settings.price)
                screen.application.invalidate()

        try:
            # 由 Agent Loop 负责模型与工具循环，界面只消费文本事件
            result = await agent_loop.run(
                session.get_messages(),
                on_event=handle_event,
                build_context=build_context,
            )
        except asyncio.CancelledError as exc:
            # 取消时保留已生成的部分回复，供下一轮继续参考
            cancelled_messages = (
                exc.new_messages if isinstance(exc, AgentLoopCancelled) else ()
            )
            _persist_new_messages(session, cancelled_messages)
            if not _persist_compactions(session, new_compactions):
                screen.set_status_message("Session persistence degraded")
            response = "".join(response_parts)
            if response and not any(
                message.role == "assistant"
                and message.content == response
                and message.status == "cancelled"
                for message in cancelled_messages
            ):
                session.add_message(
                    Message(
                        role="assistant",
                        content=response,
                        status="cancelled",
                    )
                )
            screen.append_to_entry(response_index, "(cancelled)")
            screen.commit_entry(response_index)
            raise
        except (AgentError, AgentLoopError) as exc:
            # 模型请求失败时保留部分回复和结构化错误状态
            response = "".join(response_parts)
            session.add_message(
                Message(
                    role="assistant",
                    content=response,
                    status="error",
                    error_category=(
                        exc.category if isinstance(exc, AgentError) else "internal"
                    ),
                )
            )
            screen.append_to_entry(response_index, f"Error: {exc}")
            screen.commit_entry(response_index)
        else:
            # 流式响应完成后，按 AgentLoop 返回顺序保存本轮新增消息
            _persist_new_messages(session, result.new_messages)
            if not _persist_compactions(session, new_compactions):
                screen.set_status_message("Session persistence degraded")
            if fallback_used:
                screen.add_entry(
                    "tool",
                    "⚠ Context summary failed; recent history only",
                )
            _update_persistence_status(screen, session)
            await refresh_balance()
            screen.commit_entry(response_index)
        finally:
            # 本轮请求结束（成功/失败/取消），清除 working 提示
            screen.set_working(None)

    def _render_startup_info() -> list[list[tuple[str, str]]]:
        """渲染新建会话的操作引导：命令、选择、背景图、缩放与输入框说明。"""

        parts: list[list[tuple[str, str]]] = [
            [("class:startup-hint", "type / to see commands  (/model /compact /skills /mcp …)")],
            [("class:startup-hint", "↑/↓ or mouse to select · Esc to cancel")],
            [("class:startup-hint", "/background-image to switch wallpaper · terminal controls zoom")],
            [("class:startup-hint", "input supports markdown (**bold**, `code`)")],
            [("class:startup-hint", "c-d exit")],
        ]
        return parts

    def _render_info_line() -> str:
        """渲染状态栏信息行：用量、成本、余额、上下文与压缩模式。"""

        parts: list[str] = []
        if usage_totals.prompt_tokens:
            parts.append(f"↑{format_tokens(usage_totals.prompt_tokens)}")
        if usage_totals.completion_tokens:
            parts.append(f"↓{format_tokens(usage_totals.completion_tokens)}")
        if usage_totals.cached_tokens:
            parts.append(f"R{format_tokens(usage_totals.cached_tokens)}")
        if latest_usage is not None:
            hit_rate = cache_hit_rate(latest_usage)
            if hit_rate is not None:
                parts.append(f"CH{hit_rate:.1f}%")
        if usage_totals.cost:
            parts.append(f"${usage_totals.cost:.3f}")
        parts.append(f"Balance: {current_balance}")
        estimated = context_manager.estimate_tokens(session.get_messages())
        window = context_manager.context_window
        percent = estimated / window * 100 if window else 0
        # 支持自动压缩时末尾标注 (auto)（对齐 Pi）
        parts.append(f"{percent:.1f}%/{format_tokens(window)} (auto)")
        return " ".join(parts)

    screen = ChatScreen(
        status,
        on_submit=handle_submit,
        command_names=[
            (command.name, command.description)
            for command in command_registry.list()
        ],
        model_name_provider=lambda: client_holder.settings.model_name,
        balance_text_provider=lambda: current_balance,
        provider_name_provider=lambda: infer_provider(
            client_holder.settings.base_url
        ),
        thinking_level_provider=lambda: agent_loop.thinking_level,
        info_line_provider=_render_info_line,
        copy_hint_provider=lambda: copy_hint,
        startup_info_provider=_render_startup_info,
    )
    tool_manager = ToolManager(
        permission_manager=PermissionManager(screen.request_approval),
    )
    for create_tool in (
        create_read_file_tool,
        create_list_files_tool,
        create_search_files_tool,
        create_write_file_tool,
        create_edit_file_tool,
        create_run_command_tool,
    ):
        tool_manager.register_local(*create_tool(session_workspace))
    if mcp_provider is not None:
        try:
            await tool_manager.register_mcp_provider(mcp_provider)
        except Exception:
            await mcp_provider.close()
            raise
    context_manager = ContextManager(
        context_budget or DEFAULT_CONTEXT_BUDGET,
        {
            definition.name: definition.capability
            for definition in tool_manager.list_definitions()
            if definition.capability is not None
        },
        model_tools=tool_manager.model_tools(),
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
    context_manager.set_model_name(settings.model_name)
    agent_loop = agent_loop or AgentLoop(
        client, tool_manager, max_tool_rounds=max_tool_rounds
    )

    history = session.get_messages()
    screen.add_history_entries(
        [(message.role, message.content) for message in history]
    )
    flush_history = getattr(screen, "flush_history", None)
    if flush_history is not None:
        await flush_history()
    try:
        await screen.application.run_async()
    finally:
        if mcp_provider is not None:
            await mcp_provider.close()
        if not session.close():
            screen.set_status_message("Session persistence degraded")
    return ChatExitInfo(
        session_id=None if session.deleted else session.session_id,
        usage_totals=usage_totals if latest_usage is not None else None,
    )


def _tool_call_summary(tool_call) -> str:
    """生成工具调用开始时的单行摘要。"""

    arguments = _single_line(str(tool_call.arguments), 60)
    return f"▸ {tool_call.name}  {arguments}"


def _persist_new_messages(session: Session, messages: tuple[Message, ...]) -> None:
    """将 AgentLoop 明确返回的本轮新增消息追加到 Session。"""

    for message in messages:
        session.add_message(message)


def _persist_compactions(session: Session, compactions) -> bool:
    """在本轮消息写入后追加对应压缩记录并返回持久化状态。"""

    persisted = True
    for compaction in compactions:
        persisted = session.add_compaction(compaction) and persisted
    return persisted


def _update_tool_result(
    screen,
    index: int,
    event: ToolExecutionEvent,
    success: bool = True,
) -> None:
    """展示工具结果：保留状态和名称，并由界面层统一折叠。"""

    marker = "✓" if success else "✗"
    display_content = f"{marker} {event.tool_call.name}\n{event.result.content}"
    screen.set_tool_result(index, display_content)


def _update_persistence_status(screen: ChatScreen, session: Session) -> None:
    """在持久化降级后向状态栏写入安全提示。"""

    if session.persistence_degraded:
        screen.set_status_message("Session persistence degraded")


def _single_line(content: str, limit: int) -> str:
    """压缩换行文本并限制界面摘要长度。"""

    compact = " ".join(content.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1]}…"
