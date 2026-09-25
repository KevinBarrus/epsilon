"""Loop Guard 离线回放工具的测试。"""

import json
from pathlib import Path

from core.loop_guard import LoopGuardConfig
from evaluation.replay_loop_guard import format_report, replay, tool_capabilities


def _batch(run_id: str, number: int, role: str, digest: str, call_id: str) -> dict:
    """构造一条 batch 归档记录。"""

    return {
        "type": "batch",
        "role": role,
        "agent_run_id": run_id,
        "round": number,
        "execution_mode": "sequential",
        "calls": [
            {
                "call_id": call_id,
                "tool": "read_file",
                "args_digest": digest,
                "args_preview": '{"path": "a.py"}',
            }
        ],
    }


def _result(run_id: str, call_id: str, digest: str) -> dict:
    """构造一条 result 归档记录。"""

    return {
        "type": "result",
        "role": "worker",
        "agent_run_id": run_id,
        "call_id": call_id,
        "tool": "read_file",
        "is_error": False,
        "error_category": None,
        "error_family": None,
        "args_digest": digest,
        "args_preview": '{"path": "a.py"}',
        "output_digest": "out-1",
    }


def _write(tmp_path: Path, records: list[dict]) -> Path:
    """把记录写成 child_events.jsonl。"""

    path = tmp_path / "child_events.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_replay_counts_injections_per_run(tmp_path: Path) -> None:
    """逐轮回放能给出每个 run 的注入次数、首触发轮次与信号构成。"""

    records = []
    for number in range(1, 11):
        call_id = f"c{number}"
        records.append(_batch("w1", number, "worker", "d1", call_id))
        records.append(_result("w1", call_id, "d1"))

    runs = replay(_write(tmp_path, records))
    worker = runs["w1"]

    assert worker.rounds == 10
    assert worker.injections == 4
    assert worker.first_trigger_round == 3
    assert worker.signals["repeated_call"] == 3
    assert worker.signals["no_progress"] == 1
    assert worker.longest_no_fact_streak == 9


def test_replay_skips_no_progress_for_read_only_role(tmp_path: Path) -> None:
    """Scout 不做"无新事实"检测，只剩重复调用信号。"""

    records = []
    for number in range(1, 11):
        call_id = f"c{number}"
        records.append(_batch("s1", number, "scout", "d1", call_id))
        records.append(_result("s1", call_id, "d1"))

    runs = replay(_write(tmp_path, records))
    scout = runs["s1"]

    assert scout.injections == 3
    assert "no_progress" not in scout.signals


def test_replay_groups_runs_by_agent_run_id(tmp_path: Path) -> None:
    """不同 run_id 的轨迹互不干扰。"""

    records = []
    for number in range(1, 4):
        records.append(_batch("w1", number, "worker", "d1", f"a{number}"))
        records.append(_result("w1", f"a{number}", "d1"))
        records.append(_batch("w2", number, "worker", f"e{number}", f"b{number}"))
        records.append(_result("w2", f"b{number}", f"e{number}"))

    runs = replay(_write(tmp_path, records))
    # w2 每轮都是新签名（新事实），不会触发任何信号
    assert runs["w1"].injections == 1
    assert runs["w2"].injections == 0


def test_replay_handles_old_format_archive(tmp_path: Path) -> None:
    """旧格式归档（没有 batch 记录）不报错，并明确说明无法回放。"""

    records = [_result("w1", "c1", "d1")]
    runs = replay(_write(tmp_path, records))

    assert runs == {}
    assert "无法按轮回放" in format_report(runs)


def test_replay_honrors_custom_config(tmp_path: Path) -> None:
    """回放遵循自定义阈值。"""

    records = []
    for number in range(1, 4):
        call_id = f"c{number}"
        records.append(_batch("w1", number, "worker", "d1", call_id))
        records.append(_result("w1", call_id, "d1"))

    runs = replay(_write(tmp_path, records), LoopGuardConfig(no_progress_rounds=0))
    assert runs["w1"].signals["repeated_call"] == 1


def test_tool_capabilities_matches_production_definitions() -> None:
    """回放用的 capability 表必须和真实工具定义一致，否则判定会漂移。"""

    capabilities = tool_capabilities()

    assert capabilities["read_file"] == "file.read"
    assert capabilities["list_files"] == "file.read"
    assert capabilities["search_files"] == "file.read"
    assert capabilities["write_file"] == "file.write"
    assert capabilities["edit_file"] == "file.write"
    assert capabilities["run_command"] is None
    assert capabilities["spawn_agent"] == "agent.scout"
    assert capabilities["spawn_worker"] == "agent.worker"
    assert capabilities["spawn_reviewer"] == "agent.reviewer"
