from pathlib import Path

from core.agent_loop import AgentLoopFailed
from core.errors import AgentError
from core.model import Message, ToolCall
from core.session import Session
from core.ui import _persist_new_messages


def test_ui_persists_new_tool_messages_in_agent_order(tmp_path: Path) -> None:
    """测试应用层按 AgentLoop 顺序保存工具消息。"""

    session = Session(tmp_path)
    session.add_user_message("读取文件")
    new_messages = (
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall("call-1", "read_file", {"path": "a.txt"}),),
        ),
        Message(role="tool", content="文件内容", tool_call_id="call-1"),
        Message(role="assistant", content="文件内容如下"),
    )

    _persist_new_messages(session, new_messages)

    assert session.get_messages() == [
        Message(role="user", content="读取文件"),
        *new_messages,
    ]


def test_ui_restores_cancelled_tool_chain(tmp_path: Path) -> None:
    """测试取消后的工具链可以从 Session 恢复。"""

    session = Session(tmp_path)
    session.add_user_message("读取文件")
    new_messages = (
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall("call-1", "read_file", {"path": "a.txt"}),),
            status="cancelled",
        ),
        Message(role="tool", content="文件内容", tool_call_id="call-1"),
    )
    _persist_new_messages(session, new_messages)
    assert session.flush_persistence()
    session.close()

    restored = Session.restore(tmp_path, session.session_id)

    assert restored.get_messages() == [
        Message(role="user", content="读取文件"),
        *new_messages,
    ]


def test_ui_restores_tool_chain_carried_by_model_failure(tmp_path: Path) -> None:
    """测试模型失败携带的已完成工具轨迹可以持久化并恢复。"""

    session = Session(tmp_path)
    session.add_user_message("读取文件")
    new_messages = (
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall("call-1", "read_file", {"path": "a.txt"}),),
        ),
        Message(role="tool", content="文件内容", tool_call_id="call-1"),
    )
    error = AgentLoopFailed(
        AgentError("network", "model_request", "模型请求失败"),
        new_messages,
    )
    _persist_new_messages(session, error.new_messages)
    session.add_message(
        Message(
            role="assistant",
            content="",
            status="error",
            error_category=error.category,
        )
    )
    assert session.flush_persistence()
    session.close()

    restored = Session.restore(tmp_path, session.session_id)

    assert restored.get_messages() == [
        Message(role="user", content="读取文件"),
        *new_messages,
        Message(
            role="assistant",
            content="",
            status="error",
            error_category="network",
        ),
    ]
