"""负责会话消息的 JSONL 文件读写。"""

import fcntl
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .errors import is_error_category
from .model import Message, MessageStatus, ToolCall, UsageEvent


class SessionStoreError(ValueError):
    """会话文件格式或会话标识无效时抛出的异常。"""


@dataclass(frozen=True)
class SessionSummary:
    """用于会话选择器展示的最小摘要"""

    session_id: str
    title: str
    updated_at: datetime


@dataclass(frozen=True)
class CompactionRecord:
    """记录一次上下文压缩及其原始消息保留边界。"""

    summary: str
    first_kept_message_index: int
    tokens_before: int



@dataclass(frozen=True)
class EvictedToolOutput:
    """记录一条被降级为占位符的工具结果及其取回 id。"""

    message_index: int
    artifact_id: str
    original_chars: int


@dataclass(frozen=True)
class EvictionRecord:
    """记录一次陈旧工具输出批量驱逐的边界与降级明细。

    驱逐是构建模型上下文时的视图变换：JSONL 原文永不改写，
    每次按 before_message_index 与 evicted 明细确定性回放出降级视图。
    """

    before_message_index: int
    evicted: tuple[EvictedToolOutput, ...]
    tokens_before: int


class SessionStore:
    """将一个工作区中的会话消息追加或读取为 JSONL。"""

    def __init__(self, workspace: Path) -> None:
        """记录工作区路径，不提前创建运行时目录。"""

        self._sessions_dir = workspace / ".epsilon" / "sessions"

    def acquire_session_lock(self, session_id: str) -> TextIO:
        """非阻塞获取指定 Session 的跨进程独占锁。"""

        lock_path = self._lock_path(session_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise SessionStoreError(
                f"session is already open: {session_id}"
            ) from exc
        return lock_file

    @staticmethod
    def release_session_lock(lock_file: TextIO) -> None:
        """释放 Session 独占锁并关闭锁文件。"""

        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def append_message(self, session_id: str, message: Message) -> None:
        """将一条消息追加到指定会话的 JSONL 文件。"""

        self._append_record(session_id, self._message_record(message))

    def append_pending_message(self, session_id: str, message: Message) -> None:
        """将主日志写入失败的消息追加到 pending JSONL。"""

        self._append_record(session_id, self._message_record(message), pending=True)

    def load_pending_messages(self, session_id: str) -> list[Message]:
        """读取指定会话尚未迁移到主日志的消息。"""

        path = self._pending_path(session_id)
        if not path.exists():
            return []
        messages: list[Message] = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SessionStoreError(
                        f"pending line {line_number} is not valid JSON"
                    ) from exc
                messages.append(self._message_from_record(record, line_number))
        return messages

    def clear_pending_messages(self, session_id: str) -> None:
        """删除已经迁移到主日志的 pending 文件。"""

        self._pending_path(session_id).unlink(missing_ok=True)

    def replace_pending_messages(
        self,
        session_id: str,
        messages: list[Message],
    ) -> None:
        """用尚未迁移的消息重写 pending 文件。"""

        path = self._pending_path(session_id)
        if not messages:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for message in messages:
                json.dump(self._message_record(message), file, ensure_ascii=False)
                file.write("\n")

    def append_compaction(
        self,
        session_id: str,
        compaction: CompactionRecord,
    ) -> None:
        """将一次上下文压缩记录追加到指定会话的 JSONL 文件。"""

        record = {
            "type": "compaction",
            "summary": compaction.summary,
            "first_kept_message_index": compaction.first_kept_message_index,
            "tokens_before": compaction.tokens_before,
        }
        self._append_record(session_id, record)

    def load_messages(self, session_id: str) -> list[Message]:
        """按文件顺序读取指定会话的全部消息。

        非 message 记录类型（compaction、eviction 及未来新增或未知的
        类型）一律跳过，保证旧版本会话文件可恢复。
        """

        session_path = self._session_path(session_id)
        messages: list[Message] = []
        for line_number, record in self._read_records(session_path):
            if isinstance(record, dict) and record.get("type") != "message":
                continue
            messages.append(self._message_from_record(record, line_number))
        return messages

    def load_compactions(self, session_id: str) -> list[CompactionRecord]:
        """按文件顺序读取指定会话的上下文压缩记录。"""

        session_path = self._session_path(session_id)
        compactions: list[CompactionRecord] = []
        for line_number, record in self._read_records(session_path):
            if isinstance(record, dict) and record.get("type") == "compaction":
                compactions.append(self._compaction_from_record(record, line_number))
        return compactions

    def append_eviction(self, session_id: str, eviction: EvictionRecord) -> None:
        """将一次陈旧输出驱逐记录追加到指定会话的 JSONL 文件。"""

        record = {
            "type": "eviction",
            "before_message_index": eviction.before_message_index,
            "evicted": [
                {
                    "message_index": item.message_index,
                    "artifact_id": item.artifact_id,
                    "original_chars": item.original_chars,
                }
                for item in eviction.evicted
            ],
            "tokens_before": eviction.tokens_before,
        }
        self._append_record(session_id, record)

    def load_evictions(self, session_id: str) -> list[EvictionRecord]:
        """按文件顺序读取指定会话的驱逐记录。"""

        session_path = self._session_path(session_id)
        evictions: list[EvictionRecord] = []
        for line_number, record in self._read_records(session_path):
            if isinstance(record, dict) and record.get("type") == "eviction":
                evictions.append(self._eviction_from_record(record, line_number))
        return evictions


    def list_sessions(self) -> list[SessionSummary]:
        """读取工作区中的会话摘要并按更新时间倒序排列"""

        if not self._sessions_dir.is_dir():
            return []

        summaries: list[SessionSummary] = []
        for session_path in self._sessions_dir.glob("*.jsonl"):
            session_id = session_path.stem
            try:
                title = self._read_title(session_id)
            except SessionStoreError as exc:
                raise SessionStoreError(
                    f"cannot read session {session_id}: {exc}"
                ) from exc

            summaries.append(
                SessionSummary(
                    session_id=session_id,
                    title=title,
                    updated_at=datetime.fromtimestamp(
                        session_path.stat().st_mtime,
                        tz=timezone.utc,
                    ),
                )
            )
        return sorted(summaries, key=lambda item: item.updated_at, reverse=True)

    def _read_title(self, session_id: str) -> str:
        """仅读取首条有效用户消息，避免为会话列表加载完整历史。"""

        session_path = self._session_path(session_id)
        if not session_path.exists():
            return session_id[:8]

        file_size = session_path.stat().st_size
        with session_path.open("rb") as file:
            line_number = 0
            while line := file.readline():
                line_number += 1
                line = line.decode("utf-8")
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if file.tell() == file_size and not line.endswith("\n"):
                        break
                    raise SessionStoreError(
                        "line {line_number} is not valid JSON"
                    ) from exc

                if isinstance(record, dict) and record.get("type") != "message":
                    continue
                message = self._message_from_record(record, line_number)
                if message.role == "user":
                    return self._create_title([message], session_id)
        return session_id[:8]

    def delete_session(self, session_id: str) -> bool:
        """删除会话文件（先 trash 后 unlink），返回是否删除了文件。"""

        deleted = False
        for path in (self._session_path(session_id), self._pending_path(session_id)):
            if path.exists():
                _delete_file(path)
                deleted = True
        return deleted

    def _session_path(self, session_id: str) -> Path:
        """校验 Session ID 后生成对应文件路径。"""

        try:
            normalized_id = str(uuid.UUID(session_id))
        except (ValueError, AttributeError) as exc:
            raise SessionStoreError("invalid session ID") from exc
        return self._sessions_dir / f"{normalized_id}.jsonl"

    def _pending_path(self, session_id: str) -> Path:
        """生成指定会话的 pending 文件路径。"""

        self._session_path(session_id)
        return self._sessions_dir / f".{session_id}.pending.jsonl"

    def _lock_path(self, session_id: str) -> Path:
        """生成指定会话的隐藏锁文件路径。"""

        self._session_path(session_id)
        return self._sessions_dir / f".{session_id}.lock"

    def _append_record(
        self,
        session_id: str,
        record: dict[str, object],
        *,
        pending: bool = False,
    ) -> None:
        """将一条 JSON 记录追加到主日志或 pending 日志。"""

        path = self._pending_path(session_id) if pending else self._session_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _read_records(path: Path) -> list[tuple[int, object]]:
        """读取 JSONL 记录并忽略最后一条未完成记录。"""

        if not path.exists():
            return []
        records: list[tuple[int, object]] = []
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                records.append((line_number, json.loads(line)))
            except json.JSONDecodeError as exc:
                if line_number == len(lines) and not line.endswith("\n"):
                    continue
                raise SessionStoreError(
                    "line {line_number} is not valid JSON"
                ) from exc
        return records

    @staticmethod
    def _message_record(message: Message) -> dict[str, object]:
        """将模型消息转换为 JSONL 记录。"""

        record: dict[str, object] = {
            "type": "message",
            "role": message.role,
            "content": message.content,
        }
        if message.status != "completed":
            record["status"] = message.status
        if message.error_category is not None:
            record["error_category"] = message.error_category
        if message.reasoning:
            record["reasoning"] = message.reasoning
        if message.tool_calls:
            record["tool_calls"] = [
                {
                    "call_id": tool_call.call_id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
                for tool_call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            record["tool_call_id"] = message.tool_call_id
        if message.usage is not None:
            usage_record: dict[str, object] = {
                "prompt_tokens": message.usage.prompt_tokens,
                "completion_tokens": message.usage.completion_tokens,
                "total_tokens": message.usage.total_tokens,
                "cached_tokens": message.usage.cached_tokens,
            }
            if message.usage.cache_miss_tokens is not None:
                usage_record["cache_miss_tokens"] = message.usage.cache_miss_tokens
            record["usage"] = usage_record
        if message.request_fingerprint is not None:
            record["request_fingerprint"] = message.request_fingerprint
        return record

    @staticmethod
    def _message_from_record(record: object, line_number: int) -> Message:
        """校验 JSON 记录并转换为模型消息。"""

        if not isinstance(record, dict) or record.get("type") != "message":
            raise SessionStoreError(f"line {line_number} is not a message record")

        role = record.get("role")
        content = record.get("content")
        if role not in {"user", "assistant", "tool"}:
            raise SessionStoreError(f"line {line_number} has an invalid role")
        if not isinstance(content, str):
            raise SessionStoreError(f"line {line_number} has invalid content")

        status = record.get("status", "completed")
        error_category = record.get("error_category")
        if error_category is not None and not is_error_category(error_category):
            raise SessionStoreError(f"line {line_number} has an invalid error category")
        reasoning = record.get("reasoning", "")
        if not isinstance(reasoning, str):
            raise SessionStoreError(f"line {line_number} has invalid reasoning")

        raw_tool_calls = record.get("tool_calls", [])
        if not isinstance(raw_tool_calls, list):
            raise SessionStoreError(f"line {line_number} has an invalid tool call")
        tool_calls: list[ToolCall] = []
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                raise SessionStoreError(f"line {line_number} has an invalid tool call")
            call_id = raw_tool_call.get("call_id")
            name = raw_tool_call.get("name")
            arguments = raw_tool_call.get("arguments")
            if (
                not isinstance(call_id, str)
                or not isinstance(name, str)
                or not isinstance(arguments, dict)
            ):
                raise SessionStoreError(f"line {line_number} has an invalid tool call")
            tool_calls.append(ToolCall(call_id, name, arguments))

        tool_call_id = record.get("tool_call_id")
        if tool_call_id is not None and not isinstance(tool_call_id, str):
            raise SessionStoreError(f"line {line_number} has an invalid tool call id")
        if role == "tool" and not tool_call_id:
            raise SessionStoreError(f"line {line_number} is missing a tool call id")
        usage = _usage_from_record(record.get("usage"), line_number)
        request_fingerprint = record.get("request_fingerprint")
        if request_fingerprint is not None and not isinstance(request_fingerprint, str):
            raise SessionStoreError(f"line {line_number} has an invalid request fingerprint")
        return Message(
            role=role,
            content=content,
            tool_calls=tuple(tool_calls),
            tool_call_id=tool_call_id,
            status=status,
            error_category=error_category,
            usage=usage,
            request_fingerprint=request_fingerprint,
            reasoning=reasoning,
        )

    @staticmethod
    def _compaction_from_record(
        record: object,
        line_number: int,
    ) -> CompactionRecord:
        """校验压缩记录并转换为 CompactionRecord。"""

        if not isinstance(record, dict) or record.get("type") != "compaction":
            raise SessionStoreError(f"line {line_number} is not a compaction record")

        summary = record.get("summary")
        first_kept_message_index = record.get("first_kept_message_index")
        tokens_before = record.get("tokens_before")
        if not isinstance(summary, str):
            raise SessionStoreError(f"line {line_number} has an invalid compaction summary")
        if not isinstance(first_kept_message_index, int) or first_kept_message_index < 0:
            raise SessionStoreError(f"line {line_number} has an invalid retention boundary")
        if not isinstance(tokens_before, int) or tokens_before < 0:
            raise SessionStoreError(f"line {line_number} has an invalid compaction token count")
        return CompactionRecord(
            summary=summary,
            first_kept_message_index=first_kept_message_index,
            tokens_before=tokens_before,
        )

    @staticmethod
    def _eviction_from_record(
        record: object,
        line_number: int,
    ) -> EvictionRecord:
        """校验驱逐记录并转换为 EvictionRecord。"""

        if not isinstance(record, dict) or record.get("type") != "eviction":
            raise SessionStoreError(f"line {line_number} is not an eviction record")

        before_message_index = record.get("before_message_index")
        raw_evicted = record.get("evicted")
        tokens_before = record.get("tokens_before")
        if (
            not isinstance(before_message_index, int)
            or before_message_index < 0
            or not isinstance(raw_evicted, list)
            or not isinstance(tokens_before, int)
            or tokens_before < 0
        ):
            raise SessionStoreError(f"line {line_number} has an invalid eviction record")
        evicted: list[EvictedToolOutput] = []
        for item in raw_evicted:
            if not isinstance(item, dict):
                raise SessionStoreError(f"line {line_number} has an invalid eviction record")
            message_index = item.get("message_index")
            artifact_id = item.get("artifact_id")
            original_chars = item.get("original_chars")
            if (
                not isinstance(message_index, int)
                or message_index < 0
                or not isinstance(artifact_id, str)
                or not isinstance(original_chars, int)
                or original_chars < 0
            ):
                raise SessionStoreError(f"line {line_number} has an invalid eviction record")
            evicted.append(EvictedToolOutput(message_index, artifact_id, original_chars))
        return EvictionRecord(
            before_message_index=before_message_index,
            evicted=tuple(evicted),
            tokens_before=tokens_before,
        )

    @staticmethod
    def _create_title(messages: list[Message], session_id: str) -> str:
        """从第一条用户消息生成会话标题"""

        first_user_message = next(
            (message.content for message in messages if message.role == "user"),
            "",
        )
        title = " ".join(first_user_message.split())
        if not title:
            return session_id[:8]
        if len(title) <= 40:
            return title
        return f"{title[:37]}..."


def _delete_file(path: Path) -> None:
    """先尝试系统 trash 命令，失败时永久删除文件。"""

    try:
        result = subprocess.run(
            ["trash", str(path)], capture_output=True, timeout=5
        )
        if result.returncode == 0 or not path.exists():
            return
    except (OSError, subprocess.TimeoutExpired):
        pass
    path.unlink(missing_ok=True)


def _usage_from_record(value: object, line_number: int) -> UsageEvent | None:
    """校验并恢复 assistant 消息携带的服务端 Token 用量。"""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise SessionStoreError(f"line {line_number} has invalid token usage")
    fields = (
        value.get("prompt_tokens"),
        value.get("completion_tokens"),
        value.get("total_tokens"),
    )
    cached = value.get("cached_tokens")
    cache_miss = value.get("cache_miss_tokens")
    if (
        any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in fields
        )
        or (
            cached is not None
            and (
                not isinstance(cached, int)
                or isinstance(cached, bool)
                or cached < 0
            )
        )
        or (
            cache_miss is not None
            and (
                not isinstance(cache_miss, int)
                or isinstance(cache_miss, bool)
                or cache_miss < 0
            )
        )
    ):
        raise SessionStoreError(f"line {line_number} has invalid token usage")
    return UsageEvent(fields[0], fields[1], fields[2], cached, cache_miss)
