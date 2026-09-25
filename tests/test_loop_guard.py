"""Loop Guard（空转检测与纠偏）的单元与接入测试。"""

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from core.agent_loop import AgentLoop
from core.config import ConfigError, load_settings
from core.loop_guard import (
    LoopGuardConfig,
    LoopGuardEvent,
    ToolCallLoopGuard,
)
from core.model import Message, ModelEvent, TextDelta, ToolCall, ToolCallEvent, ToolResult
from core.tools import ToolManager, create_read_file_tool

READ_FILE = "read_file"
CAPABILITIES = {
    "read_file": "file.read",
    "list_files": "file.read",
    "search_files": "file.read",
    "write_file": "file.write",
    "edit_file": "file.write",
    "run_command": None,
    "goal": None,
    "spawn_agent": "agent.scout",
}


def _call(name: str = READ_FILE, path: str = "a.py", call_id: str = "c1") -> ToolCall:
    """构造一个工具调用，默认是读同一个文件。"""

    return ToolCall(call_id=call_id, name=name, arguments={"path": path})


def _result(call_id: str = "c1", is_error: bool = False) -> ToolResult:
    """构造一个工具结果，默认成功。"""

    return ToolResult(call_id=call_id, content="ok", is_error=is_error)


def test_repeated_call_escalates_then_throttles() -> None:
    """连续相同调用在第 3/5/8 轮升级提醒，之后每 4 轮继续施压。"""

    guard = ToolCallLoopGuard()
    levels: list[tuple[int, str]] = []
    for index in range(13):
        injection = guard.observe(
            (_call(call_id=f"c{index}"),),
            (_result(f"c{index}"),),
            CAPABILITIES,
        )
        if injection is not None:
            levels.append((index + 1, injection.event.level))

    assert levels == [(3, "mild"), (5, "detailed"), (8, "detailed"), (12, "detailed")]


def test_repeated_call_counts_once_per_round() -> None:
    """同一轮里出现多次相同签名只算一次。"""

    guard = ToolCallLoopGuard()
    first = guard.observe(
        (_call(call_id="a"), _call(call_id="b")),
        (_result("a"), _result("b")),
        CAPABILITIES,
    )
    assert first is None
    assert guard.observe((_call(call_id="c"),), (_result("c"),), CAPABILITIES) is None
    third = guard.observe((_call(call_id="d"),), (_result("d"),), CAPABILITIES)
    assert third is not None and third.event.level == "mild"


def test_repeated_call_resets_when_signature_changes() -> None:
    """换成不同参数后原签名计数归零。"""

    guard = ToolCallLoopGuard()
    guard.observe((_call(path="a.py"),), (_result(),), CAPABILITIES)
    guard.observe((_call(path="a.py"),), (_result(),), CAPABILITIES)
    assert guard.observe((_call(path="b.py"),), (_result(),), CAPABILITIES) is None
    guard.observe((_call(path="b.py"),), (_result(),), CAPABILITIES)
    third = guard.observe((_call(path="b.py"),), (_result(),), CAPABILITIES)
    assert third is not None and third.event.tool_name == READ_FILE


def test_exempt_tool_is_not_tracked() -> None:
    """免检工具不计数也不提醒。"""

    guard = ToolCallLoopGuard(LoopGuardConfig(exempt_tools=(READ_FILE,)))
    for _ in range(6):
        assert guard.observe((_call(),), (_result(),), CAPABILITIES) is None


def test_failed_call_still_counts_as_repeat() -> None:
    """失败的同签名调用同样算重复（只看调用，不看结果）。"""

    guard = ToolCallLoopGuard()
    guard.observe((_call(),), (_result(is_error=True),), CAPABILITIES)
    guard.observe((_call(),), (_result(is_error=True),), CAPABILITIES)
    third = guard.observe((_call(),), (_result(is_error=True),), CAPABILITIES)
    assert third is not None


def test_progress_call_resets_repeat_counters() -> None:
    """有进展的调用会把所有签名计数清零。"""

    guard = ToolCallLoopGuard()
    guard.observe((_call(),), (_result(),), CAPABILITIES)
    guard.observe((_call(),), (_result(),), CAPABILITIES)
    assert guard.observe((_call(name="write_file"),), (_result(),), CAPABILITIES) is None
    assert guard.observe((_call(),), (_result(),), CAPABILITIES) is None
    assert guard.observe((_call(),), (_result(),), CAPABILITIES) is None
    third = guard.observe((_call(),), (_result(),), CAPABILITIES)
    assert third is not None


def test_no_progress_reminder_after_threshold() -> None:
    """连续多轮无进展时注入无进展提醒。"""

    guard = ToolCallLoopGuard(LoopGuardConfig(no_progress_rounds=8))
    for index in range(7):
        assert (
            guard.observe(
                (_call(path=f"f{index}.py"),),
                (_result(),),
                CAPABILITIES,
            )
            is None
        )
    injection = guard.observe((_call(path="f7.py"),), (_result(),), CAPABILITIES)
    assert injection is not None
    assert injection.event.kind == "no_progress"
    assert injection.event.level == "detailed"
    assert injection.event.count == 8


def test_no_progress_resets_after_progress_call() -> None:
    """出现有进展的调用后无进展计数归零。"""

    guard = ToolCallLoopGuard(LoopGuardConfig(no_progress_rounds=3))
    for index in range(2):
        guard.observe((_call(path=f"f{index}.py"),), (_result(),), CAPABILITIES)
    assert guard.observe((_call(name="run_command"),), (_result(),), CAPABILITIES) is None
    assert guard.observe((_call(path="g0.py"),), (_result(),), CAPABILITIES) is None
    assert guard.observe((_call(path="g1.py"),), (_result(),), CAPABILITIES) is None
    injection = guard.observe((_call(path="g2.py"),), (_result(),), CAPABILITIES)
    assert injection is not None and injection.event.kind == "no_progress"


def test_progress_definition_follows_capability_not_whitelist() -> None:
    """委派与命令都算进展，只有纯读文件不算。"""

    guard = ToolCallLoopGuard(LoopGuardConfig(no_progress_rounds=2))
    for index in range(6):
        delegation = guard.observe(
            (_call(name="spawn_agent", path=f"t{index}"),),
            (_result(),),
            CAPABILITIES,
        )
        assert delegation is None
        command = guard.observe(
            (_call(name="run_command", path=f"c{index}"),),
            (_result(),),
            CAPABILITIES,
        )
        assert command is None


def test_no_progress_detection_skipped_for_read_only_role() -> None:
    """只读角色（Scout）不做无进展检测。"""

    guard = ToolCallLoopGuard(LoopGuardConfig(no_progress_rounds=2), role="scout")
    for index in range(10):
        assert (
            guard.observe((_call(path=f"f{index}.py"),), (_result(),), CAPABILITIES)
            is None
        )


def test_no_progress_detection_disabled_by_zero() -> None:
    """no_progress_rounds=0 关闭检测 B。"""

    guard = ToolCallLoopGuard(LoopGuardConfig(no_progress_rounds=0))
    for index in range(10):
        assert (
            guard.observe((_call(path=f"f{index}.py"),), (_result(),), CAPABILITIES)
            is None
        )


def test_disabled_guard_never_injects() -> None:
    """enabled=False 时完全关闭。"""

    guard = ToolCallLoopGuard(LoopGuardConfig(enabled=False))
    for _ in range(10):
        assert guard.observe((_call(),), (_result(),), CAPABILITIES) is None


def test_config_rejects_invalid_values() -> None:
    """非法阈值与负数无进展轮数必须报错。"""

    with pytest.raises(ValueError):
        LoopGuardConfig(thresholds=())
    with pytest.raises(ValueError):
        LoopGuardConfig(thresholds=(3, 3))
    with pytest.raises(ValueError):
        LoopGuardConfig(thresholds=(0, 3))
    with pytest.raises(ValueError):
        LoopGuardConfig(no_progress_rounds=-1)


class RepeatReadClient:
    """前若干轮重复同一个读取调用，之后返回最终文本。"""

    def __init__(self, repeats: int = 3) -> None:
        self._repeats = repeats
        self._requests = 0

    async def stream_response(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]] = (),
        thinking_level: str | None = None,
    ) -> AsyncIterator[ModelEvent]:
        self._requests += 1
        if self._requests <= self._repeats:
            yield ToolCallEvent(
                ToolCall(
                    call_id=f"call-{self._requests}",
                    name=READ_FILE,
                    arguments={"path": "README.md"},
                )
            )
            return
        yield TextDelta("完成")


def _read_manager(tmp_path: Path) -> ToolManager:
    """注册一个真实的读文件工具，让 capability 映射可用。"""

    (tmp_path / "README.md").write_text("项目说明", encoding="utf-8")
    manager = ToolManager()
    manager.register_local(*create_read_file_tool(tmp_path))
    return manager


@pytest.mark.asyncio
async def test_agent_loop_injects_reminder_without_persisting(tmp_path: Path) -> None:
    """命中纠偏时消息只进上下文，不写入 new_messages。"""

    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    result = await AgentLoop(RepeatReadClient(), _read_manager(tmp_path)).run(
        [Message(role="user", content="读取说明")],
        on_event=collect,
    )

    assert result.loop_guard_injections == 1
    assert any(
        message.role == "system" and "重复" in message.content
        for message in result.messages
    )
    assert not any(message.role == "system" for message in result.new_messages)
    assert any(isinstance(event, LoopGuardEvent) for event in events)


@pytest.mark.asyncio
async def test_agent_loop_guard_can_be_disabled(tmp_path: Path) -> None:
    """关闭配置后不再注入纠偏。"""

    result = await AgentLoop(
        RepeatReadClient(),
        _read_manager(tmp_path),
        loop_guard_config=LoopGuardConfig(enabled=False),
    ).run([Message(role="user", content="读取说明")])

    assert result.loop_guard_injections == 0
    assert not any(message.role == "system" for message in result.messages)


@pytest.mark.asyncio
async def test_agents_hold_independent_guards(tmp_path: Path) -> None:
    """两个 AgentLoop 各自计数，互不影响。"""

    manager = _read_manager(tmp_path)
    first = await AgentLoop(RepeatReadClient(repeats=3), manager).run(
        [Message(role="user", content="一")]
    )
    second = await AgentLoop(RepeatReadClient(repeats=3), manager).run(
        [Message(role="user", content="二")]
    )

    assert first.loop_guard_injections == 1
    assert second.loop_guard_injections == 1


def _write_settings(tmp_path: Path, data: dict) -> Path:
    """写入一份用户级 settings.json。"""

    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _valid_settings() -> dict:
    """返回一份合法的模型配置。"""

    return {
        "model": {
            "base_url": "https://example.com/v1",
            "model_name": "test-model",
            "api_key": "test-key",
        }
    }


def test_settings_parses_loop_guard_object(tmp_path: Path) -> None:
    """嵌套 loop_guard 配置解析成 Settings 字段。"""

    data = _valid_settings()
    data["loop_guard"] = {
        "enabled": False,
        "thresholds": [2, 4],
        "exempt_tools": ["read_file"],
        "no_progress_rounds": 5,
    }
    settings = load_settings(user_config_path=_write_settings(tmp_path, data))

    assert settings.loop_guard_enabled is False
    assert settings.loop_guard_thresholds == (2, 4)
    assert settings.loop_guard_exempt_tools == ("read_file",)
    assert settings.loop_guard_no_progress_rounds == 5


def test_settings_loop_guard_defaults(tmp_path: Path) -> None:
    """缺省时使用默认开关。"""

    settings = load_settings(
        user_config_path=_write_settings(tmp_path, _valid_settings())
    )

    assert settings.loop_guard_enabled is True
    assert settings.loop_guard_thresholds == (3, 5, 8)
    assert settings.loop_guard_exempt_tools == ()
    assert settings.loop_guard_no_progress_rounds == 8


def test_settings_rejects_invalid_loop_guard(tmp_path: Path) -> None:
    """非法 loop_guard 配置必须报配置错误。"""

    data = _valid_settings()
    data["loop_guard"] = {"thresholds": [5, 3]}
    with pytest.raises(ConfigError):
        load_settings(user_config_path=_write_settings(tmp_path, data))


def test_loop_guard_event_is_serialisable_for_evaluation() -> None:
    """纠偏事件必须能被评测轨迹序列化，否则评测会在注入时崩溃。"""

    from evaluation.events import event_to_record

    guard = ToolCallLoopGuard()
    injection = None
    for _ in range(3):
        injection = guard.observe((_call(),), (_result(),), CAPABILITIES)
    assert injection is not None

    record = event_to_record(injection.event)
    assert record["type"] == "loop_guard"
    assert record["kind"] == "repeated_call"
    assert record["tool_name"] == READ_FILE
    assert record["count"] == 3
