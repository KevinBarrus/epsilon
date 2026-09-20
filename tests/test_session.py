import uuid
from pathlib import Path

import pytest

from core.artifacts import ArtifactStore, is_artifact_placeholder
from core.context import ContextBudget, ContextManager, _apply_evictions
from core.model import Message, ToolCall
from core.session import Session
from core.session_store import (
    CompactionRecord,
    EvictedToolOutput,
    EvictionRecord,
    SessionStore,
    SessionStoreError,
)


def test_new_sessions_have_unique_ids(tmp_path: Path) -> None:
    """测试新会话拥有合法且唯一的 Session ID"""

    first = Session(tmp_path)
    second = Session(tmp_path)

    assert uuid.UUID(first.session_id)
    assert first.session_id != second.session_id


def test_same_session_cannot_be_opened_twice(tmp_path: Path) -> None:
    """测试同一 Session 同时只能有一个写入者。"""

    session = Session(tmp_path)

    with pytest.raises(SessionStoreError, match="session is already open"):
        Session(tmp_path, session.session_id)

    session.close()


def test_closed_session_can_be_opened_again(tmp_path: Path) -> None:
    """测试关闭 Session 后会释放独占锁。"""

    first = Session(tmp_path)
    session_id = first.session_id
    first.close()

    second = Session(tmp_path, session_id)

    assert second.session_id == session_id
    second.close()


def test_different_sessions_can_be_opened_together(tmp_path: Path) -> None:
    """测试不同 Session 的锁互不影响。"""

    first = Session(tmp_path)
    second = Session(tmp_path)

    assert first.session_id != second.session_id
    first.close()
    second.close()


def test_restore_failure_releases_session_lock(tmp_path: Path) -> None:
    """测试恢复异常不会把 Session 锁永久留在当前进程。"""

    session = Session(tmp_path)
    session_id = session.session_id
    session.close()
    session_path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    session_path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(SessionStoreError):
        Session.restore(tmp_path, session_id)

    reopened = Session(tmp_path, session_id)
    reopened.close()


def test_session_updates_memory_and_store(tmp_path: Path) -> None:
    """测试追加消息会同时更新内存和 JSONL 文件"""

    session = Session(tmp_path)
    expected = [
        Message(role="user", content="你好"),
        Message(role="assistant", content="你好！"),
    ]

    session.add_user_message("你好")
    session.add_assistant_message("你好！")
    assert session.flush_persistence()

    assert session.get_messages() == expected
    assert SessionStore(tmp_path).load_messages(session.session_id) == expected


def test_session_keeps_runtime_message_when_persistence_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试 JSONL 持久化失败时运行时消息仍保留并进入降级状态。"""

    def fail_append(self, session_id: str, message: Message) -> None:
        raise OSError("JSONL 不可写入")

    monkeypatch.setattr(SessionStore, "append_message", fail_append)
    session = Session(tmp_path)
    message = Message(role="user", content="待保存消息")

    session.add_message(message)

    assert session.get_messages() == [message]
    assert not session.flush_persistence()
    assert session.persistence_degraded
    assert session.close() is False
    assert session._persistence.pending_messages == (message,)


def test_restore_recovers_message_from_pending_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试重新恢复 Session 时会迁移 pending 消息。"""

    original_append = SessionStore.append_message
    should_fail = True

    def append_message(self, session_id: str, message: Message) -> None:
        if should_fail:
            raise OSError("JSONL 不可写入")
        original_append(self, session_id, message)

    monkeypatch.setattr(SessionStore, "append_message", append_message)
    session = Session(tmp_path)
    message = Message(role="assistant", content="待恢复消息")
    session.add_message(message)
    assert not session.flush_persistence()
    assert session.close() is False

    should_fail = False
    restored = Session.restore(tmp_path, session.session_id)

    assert restored.get_messages() == [message]
    assert SessionStore(tmp_path).load_pending_messages(session.session_id) == []
    assert SessionStore(tmp_path).load_messages(session.session_id) == [message]


def test_restore_rebuilds_session_memory(tmp_path: Path) -> None:
    """测试可以从 JSONL 恢复完整的会话记忆"""

    original = Session(tmp_path)
    original.add_user_message("第一次输入")
    original.add_assistant_message("第一次回复")
    assert original.flush_persistence()
    original.close()

    restored = Session.restore(tmp_path, original.session_id)

    assert restored.session_id == original.session_id
    assert restored.get_messages() == [
        Message(role="user", content="第一次输入"),
        Message(role="assistant", content="第一次回复"),
    ]


def test_restored_session_can_continue_writing(tmp_path: Path) -> None:
    """测试恢复后的会话可以继续追加并持久化消息"""

    original = Session(tmp_path)
    original.add_user_message("之前的问题")
    assert original.flush_persistence()
    original.close()

    restored = Session.restore(tmp_path, original.session_id)
    restored.add_assistant_message("之前的回答")
    assert restored.flush_persistence()

    assert SessionStore(tmp_path).load_messages(original.session_id) == [
        Message(role="user", content="之前的问题"),
        Message(role="assistant", content="之前的回答"),
    ]


def test_session_restores_tool_messages(tmp_path: Path) -> None:
    """测试 Session 恢复后保留工具调用和工具结果。"""

    original = Session(tmp_path)
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall("call-1", "read_file", {"path": "a.txt"}),),
        ),
        Message(role="tool", content="文件内容", tool_call_id="call-1"),
    ]
    for message in messages:
        original.add_message(message)
    assert original.flush_persistence()
    original.close()

    restored = Session.restore(tmp_path, original.session_id)

    assert restored.get_messages() == messages


def test_restore_rebuilds_compaction_records(tmp_path: Path) -> None:
    """测试恢复 Session 时可以读取压缩记录"""

    original = Session(tmp_path)
    compaction = CompactionRecord("早期摘要", 1, 1200)
    SessionStore(tmp_path).append_compaction(original.session_id, compaction)
    original.close()

    restored = Session.restore(tmp_path, original.session_id)

    assert restored.get_compactions() == [compaction]


def test_session_add_compaction_updates_runtime_and_store(tmp_path: Path) -> None:
    """测试追加压缩记录时同步更新运行时状态和 JSONL"""

    session = Session(tmp_path)
    compaction = CompactionRecord("早期摘要", 1, 1200)

    session.add_compaction(compaction)
    assert session.flush_persistence()

    assert session.get_compactions() == [compaction]
    assert SessionStore(tmp_path).load_compactions(session.session_id) == [compaction]


def test_session_keeps_transient_compaction_when_message_persistence_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试消息未写入主日志时仍保留本进程压缩记录。"""

    def fail_append(self, session_id: str, message: Message) -> None:
        raise OSError("JSONL 不可写入")

    monkeypatch.setattr(SessionStore, "append_message", fail_append)
    session = Session(tmp_path)
    session.add_user_message("待保存消息")

    compaction = CompactionRecord("摘要", 0, 100)
    assert not session.add_compaction(compaction)
    assert session.get_compactions() == [compaction]
    assert session.persistence_degraded
    assert SessionStore(tmp_path).load_compactions(session.session_id) == []


def test_session_keeps_transient_compaction_when_compaction_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试压缩记录写入失败时当前进程仍可复用该记录。"""

    def fail_append(self, session_id: str, compaction: CompactionRecord) -> None:
        raise OSError("JSONL 不可写入")

    monkeypatch.setattr(SessionStore, "append_compaction", fail_append)
    session = Session(tmp_path)
    compaction = CompactionRecord("摘要", 0, 100)

    assert not session.add_compaction(compaction)
    assert session.get_compactions() == [compaction]
    assert session.persistence_degraded


class SummaryClient:
    """为恢复测试提供固定的结构化摘要。"""

    async def stream_chat(self, messages):
        yield """## Goal
目标
## Progress
进展
## Key Decisions
决策
## Next Steps
下一步
## Critical Context
上下文
"""


@pytest.mark.asyncio
async def test_restore_rebuilds_context_after_compaction(tmp_path: Path) -> None:
    """测试压缩记录持久化后，恢复会话可以重建相同模型上下文。"""

    session = Session(tmp_path)
    session.add_user_message("旧问题" + "x" * 1_000)
    session.add_assistant_message("旧回答" + "x" * 1_000)
    session.add_user_message("新问题")
    session.add_assistant_message("新回答")
    budget = ContextBudget(340, 20, 100)
    manager = ContextManager(budget)

    first_result = await manager.build_for_model_result(
        SummaryClient(), session.get_messages()
    )
    assert first_result.compaction is not None
    session.add_compaction(first_result.compaction)
    session.close()

    restored = Session.restore(tmp_path, session.session_id)
    restored_result = await manager.build_for_model_result(
        SummaryClient(), restored.get_messages(), restored.get_compactions()
    )

    assert restored_result.messages == first_result.messages
    assert restored_result.compaction is None


@pytest.mark.asyncio
async def test_restore_keeps_file_operation_sections_in_context(tmp_path: Path) -> None:
    """测试恢复会话后仍能重建文件操作摘要。"""

    session = Session(tmp_path)
    session.add_user_message("读取并修改文件")
    session.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=(
                ToolCall("read-1", "read_file", {"path": "src/app.py"}),
                ToolCall("write-1", "write_file", {"path": "src/app.py"}),
            ),
        )
    )
    session.add_message(
        Message(role="tool", content="x" * 3_000, tool_call_id="read-1")
    )
    session.add_message(Message(role="tool", content="已写入", tool_call_id="write-1"))
    session.add_assistant_message("处理完成")
    manager = ContextManager(
        ContextBudget(600, 100, 100),
        {"read_file": "file.read", "write_file": "file.write"},
    )

    first_result = await manager.build_for_model_result(
        SummaryClient(), session.get_messages()
    )
    assert first_result.compaction is not None
    session.add_compaction(first_result.compaction)
    session.close()

    restored = Session.restore(tmp_path, session.session_id)
    restored_result = await manager.build_for_model_result(
        SummaryClient(), restored.get_messages(), restored.get_compactions()
    )

    summary = restored_result.messages[0].content
    assert "<read-files>\n- src/app.py\n</read-files>" in summary
    assert "<modified-files>\n- src/app.py\n</modified-files>" in summary


def test_session_add_eviction_updates_runtime_and_store(tmp_path: Path) -> None:
    """测试追加驱逐记录时同步更新运行时状态和 JSONL"""

    session = Session(tmp_path)
    eviction = EvictionRecord(
        before_message_index=1,
        evicted=(EvictedToolOutput(0, "abcdef1234567890", 500),),
        tokens_before=800,
    )

    session.add_eviction(eviction)
    assert session.flush_persistence()

    assert session.get_evictions() == [eviction]
    assert SessionStore(tmp_path).load_evictions(session.session_id) == [eviction]


def test_restore_rebuilds_identical_eviction_view(tmp_path: Path) -> None:
    """测试驱逐记录持久化后，恢复会话可以重建相同降级视图。"""

    store = ArtifactStore(tmp_path / "artifacts")
    content = "y" * 9_000
    artifact_id = store.save(content, session_id="s-1", source_tool="run_command")
    session = Session(tmp_path)
    session.add_message(Message(role="tool", content=content, tool_call_id="c-1"))
    session.add_eviction(
        EvictionRecord(
            before_message_index=1,
            evicted=(EvictedToolOutput(0, artifact_id, len(content)),),
            tokens_before=2_000,
        )
    )
    assert session.flush_persistence()
    before = _apply_evictions(session.get_messages(), session.get_evictions(), store)
    session.close()

    restored = Session.restore(tmp_path, session.session_id)
    after = _apply_evictions(restored.get_messages(), restored.get_evictions(), store)

    assert [message.content for message in after] == [
        message.content for message in before
    ]
    assert is_artifact_placeholder(after[0].content)
