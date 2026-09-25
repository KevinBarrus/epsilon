"""Loop Guard v2（空转检测与纠偏）的单元与接入测试。"""

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from core.agent_loop import AgentLoop
from core.config import ConfigError, load_settings
from core.loop_guard import (
    ANTI_PERSISTENCE,
    LoopGuardConfig,
    LoopGuardEvent,
    ToolCallLoopGuard,
    action_digest,
    args_preview,
    error_family,
    output_fingerprint,
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


def _result(
    call_id: str = "c1",
    is_error: bool = False,
    content: str = "ok",
    error_category: str | None = None,
) -> ToolResult:
    """构造一个工具结果，默认成功。"""

    return ToolResult(call_id, content, is_error=is_error, error_category=error_category)


def _repeat_only(**overrides) -> LoopGuardConfig:
    """只保留检测 A（关闭"无新事实"检测）的配置。"""

    return LoopGuardConfig(no_progress_rounds=0, **overrides)


def _no_progress_only(**overrides) -> LoopGuardConfig:
    """只保留检测 B（把读文件排除出检测 A）的配置。"""

    return LoopGuardConfig(exempt_tools=(READ_FILE,), **overrides)


# --- 检测 A：重复调用 -------------------------------------------------------


def test_repeated_call_escalates_then_throttles() -> None:
    """连续相同调用在第 3/5/8 轮升级提醒，之后每 4 轮继续施压。"""

    guard = ToolCallLoopGuard(_repeat_only())
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

    guard = ToolCallLoopGuard(_repeat_only())
    assert (
        guard.observe(
            (_call(call_id="a"), _call(call_id="b")),
            (_result("a"), _result("b")),
            CAPABILITIES,
        )
        is None
    )
    assert guard.observe((_call(call_id="c"),), (_result("c"),), CAPABILITIES) is None
    third = guard.observe((_call(call_id="d"),), (_result("d"),), CAPABILITIES)
    assert third is not None and third.event.level == "mild"


def test_repeated_call_resets_when_signature_changes() -> None:
    """换成不同参数后原签名计数归零。"""

    guard = ToolCallLoopGuard(_repeat_only())
    guard.observe((_call(path="a.py"),), (_result(),), CAPABILITIES)
    guard.observe((_call(path="a.py"),), (_result(),), CAPABILITIES)
    assert guard.observe((_call(path="b.py"),), (_result(),), CAPABILITIES) is None
    guard.observe((_call(path="b.py"),), (_result(),), CAPABILITIES)
    third = guard.observe((_call(path="b.py"),), (_result(),), CAPABILITIES)
    assert third is not None and third.event.tool_name == READ_FILE


def test_exempt_tool_is_not_tracked() -> None:
    """免检工具不计数也不提醒。"""

    guard = ToolCallLoopGuard(_repeat_only(exempt_tools=(READ_FILE,)))
    for _ in range(6):
        assert guard.observe((_call(),), (_result(),), CAPABILITIES) is None


def test_failed_call_still_counts_as_repeat() -> None:
    """失败的同签名调用同样算重复（只看调用，不看结果）。"""

    guard = ToolCallLoopGuard(_repeat_only())
    guard.observe((_call(),), (_result(is_error=True),), CAPABILITIES)
    guard.observe((_call(),), (_result(is_error=True),), CAPABILITIES)
    assert guard.observe((_call(),), (_result(is_error=True),), CAPABILITIES) is not None


def test_disabled_guard_never_injects() -> None:
    """enabled=False 时完全关闭。"""

    guard = ToolCallLoopGuard(LoopGuardConfig(enabled=False))
    for _ in range(10):
        assert guard.observe((_call(),), (_result(),), CAPABILITIES) is None


def test_successful_write_resets_repeat_counters() -> None:
    """成功的写入会把重复计数清零。"""

    guard = ToolCallLoopGuard(_repeat_only())
    guard.observe((_call(),), (_result(),), CAPABILITIES)
    guard.observe((_call(),), (_result(),), CAPABILITIES)
    assert (
        guard.observe(
            (_call(name="write_file", path="b.txt"),),
            (_result(),),
            CAPABILITIES,
        )
        is None
    )
    assertion = guard.observe((_call(),), (_result(),), CAPABILITIES)
    assert assertion is None


# --- 检测 B：新事实语义（v2 的核心改造） ------------------------------------


def test_exploration_of_new_files_never_triggers_no_progress() -> None:
    """一直在读"新文件"的合法探索期不再误报（v1 会狂响）。"""

    guard = ToolCallLoopGuard(_no_progress_only(no_progress_rounds=8))
    for index in range(40):
        injection = guard.observe(
            (_call(path=f"f{index}.py"),),
            (_result(),),
            CAPABILITIES,
        )
        assert injection is None
    assert guard.rounds_without_new_fact == 0


def test_no_progress_triggers_when_signatures_repeat() -> None:
    """原地重读同一批文件（无新签名）时按阈值触发。"""

    guard = ToolCallLoopGuard(_no_progress_only(no_progress_rounds=4))
    injection = None
    # 第 1 轮首见签名算新事实，之后才累计无新事实轮数
    for index in range(5):
        injection = guard.observe((_call(),), (_result(),), CAPABILITIES)
    assert injection is not None
    assert injection.event.kind == "no_progress"
    assert injection.event.count == 4


def test_repeated_command_with_same_output_is_not_progress() -> None:
    """v2 修正：同命令同输出不再算进展。"""

    guard = ToolCallLoopGuard(
        _no_progress_only(no_progress_rounds=4),
    )
    injection = None
    for index in range(5):
        injection = guard.observe(
            (
                _call(name="run_command", path="pytest -q", call_id=f"c{index}"),
                _call(path="a.py", call_id=f"r{index}"),
            ),
            (_result(f"c{index}", content="3 passed"), _result(f"r{index}")),
            CAPABILITIES,
        )
    assert injection is not None and injection.event.kind == "no_progress"


def test_new_command_output_counts_as_new_fact() -> None:
    """同命令但输出变化算新事实，会重置无进展计数。"""

    guard = ToolCallLoopGuard(_no_progress_only(no_progress_rounds=3))
    for index in range(2):
        guard.observe(
            (
                _call(name="run_command", path="pytest -q", call_id=f"c{index}"),
                _call(path="a.py", call_id=f"r{index}"),
            ),
            (_result(f"c{index}", content=f"{index} failed"), _result(f"r{index}")),
            CAPABILITIES,
        )
    assert guard.rounds_without_new_fact == 0


def test_successful_write_counts_as_new_fact() -> None:
    """成功的写入本身就是新事实。"""

    guard = ToolCallLoopGuard(_no_progress_only(no_progress_rounds=2))
    for index in range(3):
        guard.observe((_call(path="a.py"),), (_result(),), CAPABILITIES)
    assert guard.rounds_without_new_fact == 2
    guard.observe(
        (_call(name="write_file", path="out.txt"),),
        (_result(),),
        CAPABILITIES,
    )
    assert guard.rounds_without_new_fact == 0


def test_delegation_counts_as_new_fact() -> None:
    """成功委派也算新事实，父 Agent 不会被误报。"""

    guard = ToolCallLoopGuard(_no_progress_only(no_progress_rounds=3))
    for index in range(6):
        injection = guard.observe(
            (_call(name="spawn_agent", path=f"t{index}"),),
            (_result(),),
            CAPABILITIES,
        )
        assert injection is None


def test_no_progress_disabled_by_zero() -> None:
    """no_progress_rounds=0 关闭检测 B。"""

    guard = ToolCallLoopGuard(LoopGuardConfig(no_progress_rounds=0, exempt_tools=(READ_FILE,)))
    for _ in range(10):
        assert guard.observe((_call(),), (_result(),), CAPABILITIES) is None


def test_no_progress_skipped_for_read_only_role() -> None:
    """只读角色（Scout）不做"无新事实"检测。"""

    guard = ToolCallLoopGuard(
        _no_progress_only(no_progress_rounds=2),
        role="scout",
    )
    for _ in range(10):
        assert guard.observe((_call(),), (_result(),), CAPABILITIES) is None


# --- 检测 A 增强：abab 与错误族 --------------------------------------------


def test_abab_cycle_is_detected() -> None:
    """A,B,A,B 交替循环会被识别。"""

    guard = ToolCallLoopGuard(_repeat_only())
    injections = []
    for index in range(8):
        path = "a.py" if index % 2 == 0 else "b.py"
        injection = guard.observe((_call(path=path),), (_result(),), CAPABILITIES)
        if injection is not None:
            injections.append((index + 1, injection.event.kind))
    assert injections and injections[0][1] == "abab_action_cycle"


def test_abab_resets_when_pattern_breaks() -> None:
    """循环被打破后不再报 abab。"""

    guard = ToolCallLoopGuard(_repeat_only())
    for index in range(3):
        path = "a.py" if index % 2 == 0 else "b.py"
        guard.observe((_call(path=path),), (_result(),), CAPABILITIES)
    guard.observe((_call(path="c.py"),), (_result(),), CAPABILITIES)
    assert guard.observe((_call(path="b.py"),), (_result(),), CAPABILITIES) is None


def test_same_error_family_triggers_across_different_calls() -> None:
    """换了调用但错误族相同，同样触发。"""

    guard = ToolCallLoopGuard(_repeat_only())
    injections = []
    for index in range(3):
        injection = guard.observe(
            (_call(name="run_command", path=f"cmd-{index}"),),
            (_result(is_error=True, content="boom", error_category="command_failed"),),
            CAPABILITIES,
        )
        if injection is not None:
            injections.append(injection.event.kind)
    assert injections == ["same_error_family"]


def test_error_family_resets_when_family_changes() -> None:
    """错误族变化后计数归零。"""

    guard = ToolCallLoopGuard(_repeat_only())
    guard.observe(
        (_call(name="run_command", path="c1"),),
        (_result(is_error=True, error_category="timeout"),),
        CAPABILITIES,
    )
    guard.observe(
        (_call(name="run_command", path="c2"),),
        (_result(is_error=True, error_category="command_failed"),),
        CAPABILITIES,
    )
    assert (
        guard.observe(
            (_call(name="run_command", path="c3"),),
            (_result(is_error=True, error_category="command_failed"),),
            CAPABILITIES,
        )
        is None
    )


# --- 每轮最多一条 + 优先级 ---------------------------------------------------


def test_at_most_one_injection_per_round_with_priority() -> None:
    """多信号同时命中时每轮只注入一条，且取最高优先级。"""

    guard = ToolCallLoopGuard(
        LoopGuardConfig(thresholds=(3, 4, 8), no_progress_rounds=3)
    )
    kinds = []
    for index in range(4):
        injection = guard.observe((_call(),), (_result(),), CAPABILITIES)
        if injection is not None:
            kinds.append((index + 1, injection.event.kind))
    assert kinds[0] == (3, "repeated_call")
    assert kinds[1] == (4, "no_progress")


def test_every_reminder_declares_temporary() -> None:
    """每条纠偏消息都要声明"临时提醒、不要持久化"。"""

    guard = ToolCallLoopGuard(LoopGuardConfig(no_progress_rounds=2))
    messages = []
    for _ in range(3):
        injection = guard.observe((_call(),), (_result(),), CAPABILITIES)
        if injection is not None:
            messages.append(injection.message.content)
    assert messages and all(ANTI_PERSISTENCE in message for message in messages)


def test_loop_guard_event_is_serialisable_for_evaluation() -> None:
    """纠偏事件必须能被评测轨迹序列化。"""

    from evaluation.events import event_to_record

    guard = ToolCallLoopGuard(_repeat_only())
    injection = None
    for _ in range(3):
        injection = guard.observe((_call(),), (_result(),), CAPABILITIES)
    assert injection is not None

    record = event_to_record(injection.event)
    assert record["type"] == "loop_guard"
    assert record["kind"] == "repeated_call"
    assert record["tool_name"] == READ_FILE
    assert record["count"] == 3


# --- 摘要与错误族口径 --------------------------------------------------------


def test_action_digest_is_stable_and_order_insensitive() -> None:
    """参数顺序不影响签名摘要，不同参数产生不同摘要。"""

    assert action_digest("t", {"a": 1, "b": 2}) == action_digest("t", {"b": 2, "a": 1})
    assert action_digest("t", {"a": 1}) != action_digest("t", {"a": 2})
    assert action_digest("t", {"a": 1}) != action_digest("u", {"a": 1})


def test_args_preview_is_bounded() -> None:
    """参数预览被截断到上限内。"""

    preview = args_preview({"command": "x" * 500})
    assert len(preview) <= 121


def test_error_family_prefers_structured_code() -> None:
    """有结构化错误码时优先用它，无码时取错误首行。"""

    assert error_family(True, "timeout", "whatever") == "timeout"
    assert error_family(True, None, "  boom happened  \nmore") == "boom happened"
    assert error_family(False, "timeout", "ok") is None
    assert error_family(True, None, "   ") == "unknown_error"


def test_output_fingerprint_changes_with_content() -> None:
    """输出指纹随内容变化。"""

    assert output_fingerprint("a") == output_fingerprint("a")
    assert output_fingerprint("a") != output_fingerprint("b")


# --- 配置解析 ---------------------------------------------------------------


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


# --- AgentLoop 接入 ---------------------------------------------------------


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


@pytest.mark.asyncio
async def test_loop_guard_event_carries_run_id(tmp_path: Path) -> None:
    """纠偏事件也带 run_id，便于按 Worker 归因与回放。"""

    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    await AgentLoop(
        RepeatReadClient(repeats=3),
        _read_manager(tmp_path),
        run_id="worker-9",
        agent_role="worker",
    ).run([Message(role="user", content="读取说明")], on_event=collect)

    guard_events = [event for event in events if isinstance(event, LoopGuardEvent)]
    assert guard_events
    assert all(event.agent_run_id == "worker-9" for event in guard_events)
    assert all(event.agent_role == "worker" for event in guard_events)


@pytest.mark.asyncio
async def test_agent_loop_events_carry_run_id(tmp_path: Path) -> None:
    """事件带上 run_id，供离线按 Worker 归因。"""

    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    await AgentLoop(
        RepeatReadClient(repeats=1),
        _read_manager(tmp_path),
        run_id="worker-1",
        agent_role="worker",
    ).run([Message(role="user", content="读取说明")], on_event=collect)

    from core.agent_loop import ToolBatchEvent, ToolExecutionEvent

    executions = [event for event in events if isinstance(event, ToolExecutionEvent)]
    batches = [event for event in events if isinstance(event, ToolBatchEvent)]
    assert executions and all(event.agent_run_id == "worker-1" for event in executions)
    assert batches and all(event.agent_run_id == "worker-1" for event in batches)


def test_settings_parses_completion_gate_object(tmp_path: Path) -> None:
    """嵌套 completion_gate 配置解析成 Settings 字段。"""

    data = _valid_settings()
    data["completion_gate"] = {
        "enabled": False,
        "max_rejections": 5,
        "verifier_timeout_seconds": 30,
        "verifier_token_budget": 1000,
        "verifier_thinking": "low",
        "no_tool_nudge_rounds": 2,
    }
    settings = load_settings(user_config_path=_write_settings(tmp_path, data))

    assert settings.completion_gate_enabled is False
    assert settings.completion_gate_max_rejections == 5
    assert settings.completion_gate_verifier_timeout_seconds == 30
    assert settings.completion_gate_verifier_token_budget == 1000
    assert settings.completion_gate_verifier_thinking == "low"
    assert settings.completion_gate_no_tool_nudge_rounds == 2


def test_settings_completion_gate_defaults(tmp_path: Path) -> None:
    """缺省时完成门默认开启，且超时默认 300 秒、验证器预算默认 2M。"""

    settings = load_settings(_write_settings(tmp_path, _valid_settings()))

    assert settings.completion_gate_enabled is True
    assert settings.completion_gate_max_rejections == 2
    assert settings.completion_gate_verifier_timeout_seconds == 300.0
    assert settings.completion_gate_verifier_token_budget == 2_000_000
    assert settings.completion_gate_verifier_thinking == "high"
    assert settings.completion_gate_no_tool_nudge_rounds == 3


def test_settings_rejects_invalid_completion_gate(tmp_path: Path) -> None:
    """非法 completion_gate 配置必须报配置错误。"""

    data = _valid_settings()
    data["completion_gate"] = {"verifier_token_budget": 0}
    with pytest.raises(ConfigError):
        load_settings(user_config_path=_write_settings(tmp_path, data))
