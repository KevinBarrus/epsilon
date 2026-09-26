"""完成门 v2 的证据包与脚本化检查器测试。"""

from pathlib import Path

import pytest

from core.completion_evidence import (
    MAX_BRIEF_CHARS,
    MAX_CHANGE_ITEMS,
    BoundedReadGuard,
    EvidenceBundle,
    CheckResult,
    extract_changes,
    recent_tail,
    run_checks,
)
from core.goal import Goal, GoalPolicy
from core.model import Message, ToolCall, ToolResult


# --- 1. 证据包必须有界 -------------------------------------------------------


def test_evidence_brief_is_bounded_on_huge_inputs() -> None:
    """超长输入下整包仍 ≤ 14,000 字符。"""

    bundle = EvidenceBundle(
        objective="目" * 100_000,
        criteria="标" * 100_000,
        claim="称" * 100_000,
        changes=tuple(f"ok write_file f{i}.py" for i in range(5_000)),
        commands=tuple(f"ok exit=0 cmd{i}" for i in range(5_000)),
        checks=(CheckResult("path_exists", True, "x" * 100_000),),
        tail=tuple("尾巴" * 5_000 for _ in range(50)),
        unscripted=("条" * 10_000,),
    )

    brief = bundle.render()

    assert len(brief) <= MAX_BRIEF_CHARS
    assert "目标" in brief  # 高优先级内容保留


def test_evidence_brief_trims_low_priority_first() -> None:
    """超限时先裁低优先级小节（尾巴/命令/改动），保留目标与验收标准。"""

    bundle = EvidenceBundle(
        objective="目标内容" + "补" * 4_000,
        criteria="验收标准内容" + "补" * 4_000,
        claim="声称内容" + "补" * 2_000,
        changes=tuple(f"ok write_file f{i}.py" for i in range(500)),
        commands=tuple(f"ok exit=0 cmd{i}" for i in range(500)),
        checks=(CheckResult("path_exists", True, "补" * 4_000),),
        tail=tuple("很长的尾巴" * 200 for _ in range(10)),
    )

    brief = bundle.render()

    assert len(brief) <= MAX_BRIEF_CHARS
    assert "目标内容" in brief and "验收标准内容" in brief
    assert "近期对话尾巴" not in brief


# --- 2. 改动与命令来自动作记录 ----------------------------------------------


def test_extract_changes_reads_action_stream() -> None:
    """从工具调用与结果里提取改动文件与命令，并标注成功/失败。"""

    calls = (
        ToolCall("c1", "write_file", {"path": "a.py"}),
        ToolCall("c2", "edit_file", {"path": "b.py"}),
        ToolCall("c3", "run_command", {"command": "pytest -q"}),
        ToolCall("c4", "read_file", {"path": "c.py"}),
    )
    results = (
        ToolResult("c1", "ok"),
        ToolResult("c2", "rejected", is_error=True),
        ToolResult("c3", "exit code: 1\n1 failed", is_error=True),
        ToolResult("c4", "内容"),
    )

    changes, commands = extract_changes(calls, results)

    assert changes == ("ok write_file a.py", "failed edit_file b.py")
    assert commands == ("failed exit=1 pytest -q",), commands
    assert all("read_file" not in item for item in changes)


def test_extract_changes_is_capped_at_hundred_items() -> None:
    """改动/命令清单各自最多 100 条。"""

    calls = tuple(ToolCall(f"c{i}", "write_file", {"path": f"f{i}.py"}) for i in range(500))
    results = tuple(ToolResult(f"c{i}", "ok") for i in range(500))

    changes, _ = extract_changes(calls, results)

    assert len(changes) == MAX_CHANGE_ITEMS


def test_recent_tail_keeps_last_messages() -> None:
    """近期尾巴只取最后若干条消息。"""

    messages = tuple(Message(role="user", content=f"m{i}") for i in range(10))

    tail = recent_tail(messages)

    assert len(tail) == 5
    assert "m9" in tail[-1]


# --- 3. 脚本化检查器 --------------------------------------------------------


def _workspace(tmp_path: Path) -> Path:
    """建一个带交付物与源码的最小工作区。"""

    (tmp_path / "src").mkdir()
    (tmp_path / "src/real.py").write_text("x", encoding="utf-8")
    (tmp_path / "report.md").write_text(
        "# 子系统 A\n见 src/real.py 与 src/ghost.py\n\n# 子系统 B\n只写了一句说明\n",
        encoding="utf-8",
    )
    return tmp_path


def test_check_path_exists(tmp_path: Path) -> None:
    """path_exists：真实路径通过，编造的路径不通过。"""

    workspace = _workspace(tmp_path)

    ok = run_checks([{"kind": "path_exists", "paths": ["src/real.py"]}], workspace)
    bad = run_checks([{"kind": "path_exists", "paths": ["src/ghost.py"]}], workspace)

    assert ok[0].passed and not bad[0].passed
    assert "ghost" in bad[0].detail


def test_check_sections_cover(tmp_path: Path) -> None:
    """sections_cover：缺小节判失败。"""

    workspace = _workspace(tmp_path)

    ok = run_checks(
        [{"kind": "sections_cover", "path": "report.md", "sections": ["# 子系统 A", "# 子系统 B"]}],
        workspace,
    )
    bad = run_checks(
        [{"kind": "sections_cover", "path": "report.md", "sections": ["# 子系统 C"]}], workspace
    )

    assert ok[0].passed and not bad[0].passed


def test_check_paths_per_section(tmp_path: Path) -> None:
    """paths_per_section：每节必须有 ≥N 个真实存在的路径。"""

    workspace = _workspace(tmp_path)

    ok = run_checks(
        [
            {
                "kind": "paths_per_section",
                "path": "report.md",
                "sections": ["# 子系统 A"],
                "min_paths": 1,
            }
        ],
        workspace,
    )
    bad = run_checks(
        [
            {
                "kind": "paths_per_section",
                "path": "report.md",
                "sections": ["# 子系统 B"],
                "min_paths": 1,
            }
        ],
        workspace,
    )

    assert ok[0].passed and not bad[0].passed


def test_check_files_changed(tmp_path: Path) -> None:
    """files_changed：依据动作记录判断文件是否真的被改过。"""

    workspace = _workspace(tmp_path)
    changes = ("ok write_file src/real.py",)

    ok = run_checks(
        [{"kind": "files_changed", "paths": ["src/real.py"]}], workspace, changes
    )
    bad = run_checks(
        [{"kind": "files_changed", "paths": ["src/other.py"]}], workspace, changes
    )

    assert ok[0].passed and not bad[0].passed


def test_check_command_passed(tmp_path: Path) -> None:
    """command_passed：退出码为 0 才通过。"""

    workspace = _workspace(tmp_path)

    ok = run_checks([{"kind": "command_passed", "command": "true"}], workspace)
    bad = run_checks([{"kind": "command_passed", "command": "false"}], workspace)

    assert ok[0].passed and not bad[0].passed
    assert "exit=1" in bad[0].detail


def test_unknown_checker_does_not_pretend_to_pass(tmp_path: Path) -> None:
    """不认识的检查器不许假装通过。"""

    result = run_checks([{"kind": "magic"}], _workspace(tmp_path))

    assert not result[0].passed


# --- 4. 无脚本证据 → inconclusive -------------------------------------------


def test_unscripted_criteria_are_marked_in_brief(tmp_path: Path) -> None:
    """没有脚本证据的条目要在证据包里明确标注。"""

    goal = Goal("写报告", acceptance_criteria="报告必须很有洞察力")
    policy = GoalPolicy(goal, verifier=lambda brief: None, workspace=tmp_path, checks=())

    brief = policy._assemble_brief()

    assert "无脚本证据" in brief
    assert "报告必须很有洞察力" in brief


@pytest.mark.asyncio
async def test_unscripted_criteria_cannot_be_met(tmp_path: Path) -> None:
    """无脚本证据的条目不得被判 met：验证器按其提示词应回 inconclusive。"""

    async def honest_verifier(brief: str) -> str:
        # 模拟"按提示词只在有脚本证据时才判 met"
        return "VERDICT: met" if "[PASS]" in brief else "VERDICT: inconclusive"

    goal = Goal("写报告", acceptance_criteria="报告必须很有洞察力")
    policy = GoalPolicy(
        goal, verifier=honest_verifier, workspace=tmp_path, checks=(), max_rejections=1
    )

    outcome = await policy.request_completion()

    assert not outcome.accepted
    assert outcome.outcome == "inconclusive"


# --- 6. subagent 模式的有界读 -----------------------------------------------


def test_bounded_read_guard_limits_files_and_lines() -> None:
    """可选的 subagent 验证模式：最多 3 个文件、每个最多 200 行。"""

    guard = BoundedReadGuard()

    assert guard.allow("a.py") is None
    assert guard.allow("b.py") is None
    assert guard.allow("c.py") is None
    assert guard.allow("d.py") is not None, "第 4 个文件必须被拒"
    assert guard.allow("a.py") is None, "重复读同一个文件不算新文件"

    content = "\n".join(f"line{i}" for i in range(500))
    clipped, was_clipped = guard.clip(content)

    assert was_clipped
    assert len(clipped.splitlines()) == 200


# --- 5. evaluator 模式不给工具 ----------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_mode_passes_no_tools() -> None:
    """evaluator 模式下验证器拿不到任何工具。"""

    from evaluation.read_summary_compare import judge_completion

    seen: list[list[object]] = []

    class FakeClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            seen.append(list(tools))
            from core.model import TextDelta

            yield TextDelta("VERDICT: met")

    text = await judge_completion(FakeClient(), "证据简报")

    assert text == "VERDICT: met"
    assert seen == [[]], "evaluator 模式不得给工具"


# --- 验收 #3：门的反馈不得泄露隐藏清单 --------------------------------------


def test_gate_feedback_does_not_leak_hidden_checklist() -> None:
    """门的反馈文本里不得出现任何隐藏清单条目的关键词。

    门的反馈会回注给模型（"你缺 X"），所以评测 criteria 必须泛化；
    隐藏清单要继续当"模型看不见的测量仪器"。
    """

    from core.completion_evidence import EvidenceBundle
    from core.goal import verifier_prompt
    from evaluation.big_repo_read_score import SUBSYSTEM_CHECKLIST
    from evaluation.read_summary_score import CHECKLIST

    keywords = [k for item in CHECKLIST for k in item.keywords]
    keywords += [
        k for items in SUBSYSTEM_CHECKLIST.values() for item in items for k in item.keywords
    ]

    brief = EvidenceBundle(
        objective="通读仓库并写出实现剖析报告",
        criteria="报告必须为每个核心子系统各写一段剖析，并标注真实代码位置",
        claim="我已经完成了任务",
    ).render()
    feedback = (verifier_prompt(brief) + "\n" + brief).lower()

    leaked = sorted(k for k in keywords if k.lower() in feedback)
    assert leaked == [], f"门反馈泄露了隐藏清单关键词：{leaked}"
