"""管理可持久化的单个 Coding Agent 会话"""

import uuid
from pathlib import Path

from .memory import Memory
from .model import Message
from .session_persistence import SessionPersistenceQueue
from .session_store import CompactionRecord, SessionStore


class Session:
    """组合会话标识、运行时记忆和 JSONL 存储"""

    def __init__(self, workspace: Path, session_id: str | None = None) -> None:
        """创建一个新会话，不读取已有文件"""

        self.workspace = workspace
        self.session_id = session_id or str(uuid.uuid4())
        self._memory = Memory()
        self._store = SessionStore(workspace)
        self._compactions: list[CompactionRecord] = []
        self._transient_compactions: list[CompactionRecord] = []
        self._compaction_persistence_degraded = False
        self._deleted = False
        self._persistence = SessionPersistenceQueue(
            lambda message: self._store.append_message(self.session_id, message),
            persist_pending=lambda message: self._store.append_pending_message(
                self.session_id, message
            ),
        )

    @classmethod
    def restore(cls, workspace: Path, session_id: str) -> "Session":
        """从已有 JSONL 文件恢复一个会话"""

        session = cls(workspace, session_id)
        messages = session._store.load_messages(session.session_id)
        pending_messages = session._store.load_pending_messages(session.session_id)
        recovered_count = 0
        for message in pending_messages:
            try:
                session._store.append_message(session.session_id, message)
            except Exception:
                break
            recovered_count += 1
        session._store.replace_pending_messages(
            session.session_id,
            pending_messages[recovered_count:],
        )
        for message in [*messages, *pending_messages]:
            session._add_to_memory(message)
        session._compactions = session._store.load_compactions(session.session_id)
        return session

    def add_user_message(self, content: str) -> None:
        """持久化并追加一条用户消息"""

        self._append_message(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """持久化并追加一条模型消息"""

        self._append_message(Message(role="assistant", content=content))

    def add_message(self, message: Message) -> None:
        """持久化并追加一条完整消息"""

        self._append_message(message)

    def get_messages(self) -> list[Message]:
        """返回当前会话的消息历史"""

        return self._memory.get_messages()

    def get_compactions(self) -> list[CompactionRecord]:
        """返回当前会话的压缩记录副本"""

        return [*self._compactions, *self._transient_compactions]

    def add_compaction(self, compaction: CompactionRecord) -> bool:
        """持久化压缩记录，失败时仅保留当前进程内状态。"""

        if not self._persistence.flush():
            return self._retain_transient_compaction(compaction)
        try:
            self._store.append_compaction(self.session_id, compaction)
        except OSError:
            return self._retain_transient_compaction(compaction)
        self._compactions.append(compaction)
        return True

    def flush_persistence(self) -> bool:
        """等待当前消息队列写入完成并返回持久化状态"""

        return self._persistence.flush()

    def close(self) -> bool:
        """刷新并关闭当前会话的持久化队列"""

        return self._persistence.close()

    def mark_deleted(self) -> None:
        """标记当前会话已删除，供退出时省略恢复指引"""

        self._deleted = True

    @property
    def deleted(self) -> bool:
        """返回当前会话是否已由用户删除"""

        return self._deleted

    @property
    def persistence_degraded(self) -> bool:
        """返回当前会话是否出现持久化降级"""

        return self._persistence.degraded or self._compaction_persistence_degraded

    def _append_message(self, message: Message) -> None:
        """先更新运行时记忆，再交给后台队列持久化"""

        self._add_to_memory(message)
        self._persistence.enqueue(message)

    def _add_to_memory(self, message: Message) -> None:
        """将已有消息按角色追加到运行时记忆"""

        self._memory.add_message(message)

    def _retain_transient_compaction(self, compaction: CompactionRecord) -> bool:
        """保留未落盘压缩记录，供当前进程后续请求复用。"""

        self._transient_compactions.append(compaction)
        self._compaction_persistence_degraded = True
        return False
