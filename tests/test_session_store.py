import json
import os
import uuid
from pathlib import Path

import pytest

from core.model import Message, ToolCall, UsageEvent
from core.session_store import CompactionRecord, SessionStore, SessionStoreError


def test_append_creates_session_file_and_directory(tmp_path: Path) -> None:
    """测试第一次追加消息时才创建存储目录和文件。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())

    store.append_message(session_id, Message(role="user", content="你好"))

    session_path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    assert session_path.exists()
    assert session_path.parent.is_dir()


def test_jsonl_contains_one_json_record_per_line(tmp_path: Path) -> None:
    """测试多条消息按顺序写成独立的 JSON 记录。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    store.append_message(session_id, Message(role="user", content="你好"))
    store.append_message(session_id, Message(role="assistant", content="你好！"))

    path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert records == [
        {"type": "message", "role": "user", "content": "你好"},
        {"type": "message", "role": "assistant", "content": "你好！"},
    ]


def test_load_messages_restores_history_in_order(tmp_path: Path) -> None:
    """测试 JSONL 可以恢复为有序的 Message 列表。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    expected = [
        Message(role="user", content="你好"),
        Message(role="assistant", content="你好！"),
    ]
    for message in expected:
        store.append_message(session_id, message)

    assert store.load_messages(session_id) == expected


def test_jsonl_persists_assistant_usage_anchor(tmp_path: Path) -> None:
    """测试服务端 usage 和请求指纹可跨进程恢复。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    expected = Message(
        role="assistant",
        content="完成",
        usage=UsageEvent(120, 8, 128, 20),
        request_fingerprint="request-hash",
    )

    store.append_message(session_id, expected)

    assert store.load_messages(session_id) == [expected]


def test_load_messages_ignores_incomplete_tail_record(tmp_path: Path) -> None:
    """测试最后一条未完成 JSONL 记录不会阻止历史恢复。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"type":"message","role":"user","content":"历史"}\n'
        '{"type":"message","role":"assistant","content":"未完成',
        encoding="utf-8",
    )

    assert store.load_messages(session_id) == [
        Message(role="user", content="历史")
    ]


def test_load_compactions_ignores_incomplete_tail_record(tmp_path: Path) -> None:
    """测试压缩记录文件最后一条未完成记录可以被忽略。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"type":"compaction","summary":"摘要",'
        '"first_kept_message_index":1,"tokens_before":10}\n'
        '{"type":"compaction","summary":"未完成',
        encoding="utf-8",
    )

    assert store.load_compactions(session_id) == [
        CompactionRecord("摘要", 1, 10)
    ]


def test_compaction_record_is_persisted_and_loaded_separately(tmp_path: Path) -> None:
    """测试压缩记录与消息记录共存但分开读取。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    expected = CompactionRecord("早期历史摘要", 3, 1200)

    store.append_message(session_id, Message(role="user", content="最新消息"))
    store.append_compaction(session_id, expected)

    assert store.load_messages(session_id) == [
        Message(role="user", content="最新消息")
    ]
    assert store.load_compactions(session_id) == [expected]


@pytest.mark.parametrize(
    ("record", "expected_message"),
    [
        (
            '{"type":"compaction","summary":1,"first_kept_message_index":0,"tokens_before":1}',
            "has an invalid compaction summary",
        ),
        (
            '{"type":"compaction","summary":"摘要","first_kept_message_index":-1,"tokens_before":1}',
            "has an invalid retention boundary",
        ),
        (
            '{"type":"compaction","summary":"摘要","first_kept_message_index":0,"tokens_before":-1}',
            "has an invalid compaction token count",
        ),
    ],
)
def test_invalid_compaction_record_raises_clear_error(
    tmp_path: Path,
    record: str,
    expected_message: str,
) -> None:
    """测试损坏的压缩记录不会被静默忽略。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(record + "\n", encoding="utf-8")

    with pytest.raises(SessionStoreError, match=expected_message):
        store.load_compactions(session_id)


def test_jsonl_persists_tool_calls_and_results(tmp_path: Path) -> None:
    """测试 JSONL 可以保存和恢复工具调用及结果。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    expected = [
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall("call-1", "read_file", {"path": "a.txt"}),),
        ),
        Message(role="tool", content="文件内容", tool_call_id="call-1"),
    ]

    for message in expected:
        store.append_message(session_id, message)

    assert store.load_messages(session_id) == expected


def test_jsonl_persists_assistant_error_status(tmp_path: Path) -> None:
    """测试 JSONL 可以保存和恢复 assistant 的异常状态。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    expected = Message(
        role="assistant",
        content="部分回复",
        status="error",
        error_category="network",
    )

    store.append_message(session_id, expected)

    path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "error"
    assert record["error_category"] == "network"
    assert store.load_messages(session_id) == [expected]


def test_missing_session_returns_empty_history(tmp_path: Path) -> None:
    """测试不存在的会话文件返回空历史。"""

    assert SessionStore(tmp_path).load_messages(str(uuid.uuid4())) == []


def test_sessions_are_stored_in_separate_files(tmp_path: Path) -> None:
    """测试不同会话不会共用消息文件。"""

    store = SessionStore(tmp_path)
    first_id = str(uuid.uuid4())
    second_id = str(uuid.uuid4())
    store.append_message(first_id, Message(role="user", content="第一会话"))
    store.append_message(second_id, Message(role="user", content="第二会话"))

    assert store.load_messages(first_id) == [
        Message(role="user", content="第一会话")
    ]
    assert store.load_messages(second_id) == [
        Message(role="user", content="第二会话")
    ]


@pytest.mark.parametrize(
    ("content", "expected_message"),
    [
        ("not-json", "is not valid JSON"),
        ('{"type":"message","role":"system","content":"x"}', "has an invalid role"),
        ('{"type":"message","role":"user","content":1}', "has invalid content"),
    ],
)
def test_invalid_records_raise_clear_errors(
    tmp_path: Path,
    content: str,
    expected_message: str,
) -> None:
    """测试损坏的消息记录不会被静默忽略。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(content + "\n", encoding="utf-8")

    with pytest.raises(SessionStoreError, match=expected_message):
        store.load_messages(session_id)


def test_unknown_record_types_are_skipped_on_restore(tmp_path: Path) -> None:
    """测试未知记录类型跳过不报错，消息恢复不受影响。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"type":"message","role":"user","content":"问题"}\n'
        '{"type":"other","payload":{}}\n'
        '{"type":"message","role":"assistant","content":"回答"}\n',
        encoding="utf-8",
    )

    assert store.load_messages(session_id) == [
        Message(role="user", content="问题"),
        Message(role="assistant", content="回答"),
    ]


def test_invalid_session_id_is_rejected(tmp_path: Path) -> None:
    """测试无效 Session ID 不会生成越界文件路径。"""

    store = SessionStore(tmp_path)

    with pytest.raises(SessionStoreError, match="invalid session ID"):
        store.load_messages("../../outside")


def test_list_sessions_returns_titles_and_recent_first(tmp_path: Path) -> None:
    """测试会话摘要包含标题并按更新时间倒序排列"""

    store = SessionStore(tmp_path)
    older_id = str(uuid.uuid4())
    newer_id = str(uuid.uuid4())
    store.append_message(older_id, Message(role="user", content="旧会话"))
    store.append_message(newer_id, Message(role="user", content="新会话"))

    older_path = tmp_path / ".epsilon" / "sessions" / f"{older_id}.jsonl"
    newer_path = tmp_path / ".epsilon" / "sessions" / f"{newer_id}.jsonl"
    older_time = newer_path.stat().st_mtime - 10
    older_path.touch()
    newer_path.touch()
    os.utime(older_path, (older_time, older_time))

    summaries = store.list_sessions()

    assert [summary.session_id for summary in summaries] == [newer_id, older_id]
    assert summaries[0].title == "新会话"
    assert summaries[1].title == "旧会话"
    assert summaries[0].updated_at > summaries[1].updated_at


def test_list_sessions_truncates_title_and_uses_id_for_empty_session(
    tmp_path: Path,
) -> None:
    """测试标题会清理换行并限制长度，空会话使用 Session ID"""

    store = SessionStore(tmp_path)
    long_id = str(uuid.uuid4())
    empty_id = str(uuid.uuid4())
    long_title = "这是一个很长的会话标题\n" + "内容" * 30
    store.append_message(long_id, Message(role="user", content=long_title))
    empty_path = tmp_path / ".epsilon" / "sessions" / f"{empty_id}.jsonl"
    empty_path.touch()

    summaries = {
        summary.session_id: summary for summary in store.list_sessions()
    }

    assert len(summaries[long_id].title) == 40
    assert "\n" not in summaries[long_id].title
    assert summaries[empty_id].title == empty_id[:8]


def test_list_sessions_reads_only_until_first_user_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试会话列表不会为标题加载首条用户消息之后的历史。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"type":"message","role":"user","content":"首条标题"}\n'
        'not-json\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        store,
        "load_messages",
        lambda session_id: pytest.fail("会话列表不应加载完整历史"),
    )

    summaries = store.list_sessions()

    assert summaries[0].title == "首条标题"


def test_list_sessions_ignores_incomplete_tail_before_any_user_message(
    tmp_path: Path,
) -> None:
    """测试标题读取沿用最后一条未完成记录的恢复规则。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"type":"message","role":"assistant","content":"历史"}\n'
        '{"type":"message","role":"user","content":"未完成',
        encoding="utf-8",
    )

    assert store.list_sessions()[0].title == session_id[:8]


def test_list_sessions_rejects_corrupted_file(tmp_path: Path) -> None:
    """测试损坏的会话文件会返回明确错误"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(SessionStoreError, match=f"cannot read session {session_id}"):
        store.list_sessions()


def test_delete_session_removes_file(tmp_path: Path) -> None:
    """测试删除会话会移除 JSONL 文件。"""

    from core.session_store import SessionStore

    store = SessionStore(tmp_path)
    session_id = "11111111-1111-1111-1111-111111111111"
    store.append_message(session_id, Message(role="user", content="你好"))

    session_path = store._session_path(session_id)
    assert session_path.exists()

    assert store.delete_session(session_id) is True
    assert not session_path.exists()


def test_delete_session_missing_returns_false(tmp_path: Path) -> None:
    """测试删除不存在的会话返回 False。"""

    from core.session_store import SessionStore

    store = SessionStore(tmp_path)

    assert (
        store.delete_session("22222222-2222-2222-2222-222222222222") is False
    )


def test_jsonl_persists_assistant_reasoning(tmp_path: Path) -> None:
    """测试 assistant 思考内容可跨进程恢复。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    expected = Message(role="assistant", content="完成", reasoning="先分析任务")

    store.append_message(session_id, expected)

    assert store.load_messages(session_id) == [expected]


def test_legacy_record_without_reasoning_restores_empty(tmp_path: Path) -> None:
    """测试旧格式记录缺少 reasoning 字段时默认恢复为空。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"type":"message","role":"assistant","content":"历史回复"}\n',
        encoding="utf-8",
    )

    assert store.load_messages(session_id) == [
        Message(role="assistant", content="历史回复", reasoning="")
    ]


def test_invalid_reasoning_record_raises_clear_error(tmp_path: Path) -> None:
    """测试 reasoning 字段类型非法时报明确错误。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    path = tmp_path / ".epsilon" / "sessions" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"type":"message","role":"assistant","content":"回复","reasoning":42}\n',
        encoding="utf-8",
    )

    with pytest.raises(SessionStoreError, match="invalid reasoning"):
        store.load_messages(session_id)


def test_jsonl_persists_cache_miss_tokens(tmp_path: Path) -> None:
    """测试 usage 的缓存未命中字段可跨进程恢复，旧记录缺失时为 None。"""

    store = SessionStore(tmp_path)
    session_id = str(uuid.uuid4())
    expected = Message(
        role="assistant",
        content="完成",
        usage=UsageEvent(120, 8, 128, 20, 100),
    )

    store.append_message(session_id, expected)

    assert store.load_messages(session_id) == [expected]
