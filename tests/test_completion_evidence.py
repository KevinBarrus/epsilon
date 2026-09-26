"""完成门 v2 的证据包与脚本化检查器测试。"""

from collections.abc import Sequence
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


def test_paths_per_section_ignores_preamble_mention(tmp_path: Path) -> None:
    """检查器必须认标题行，不能把前言里提到的同名单词当成小节正文。"""

    workspace = tmp_path
    (workspace / "src").mkdir()
    (workspace / "src/real.py").write_text("x", encoding="utf-8")
    (workspace / "report.md").write_text(
        "本报告覆盖 protocol、config 两个子系统。\n\n# protocol\n见 src/real.py 与 src/real.py\n",
        encoding="utf-8",
    )

    result = run_checks(
        [{"kind": "paths_per_section", "path": "report.md", "sections": ["protocol"], "min_paths": 1}],
        workspace,
    )

    assert result[0].passed, result[0].detail


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


# ===================== v3：子系统问题模板与标识符存在性核验 =====================

from core.completion_evidence import (  # noqa: E402
    QUESTION_TYPES,
    CorpusIndex,
    classify_identifier,
    subsystem_checks,
    subsystem_criteria_text,
    validate_checks,
)

# 语料里的真实标识符：常量 / 类型 / 函数各一（刻意用不同类别，便于测"类别不全"）
CORPUS = """
pub const MAX_STDOUT_BYTES: usize = 1024;
pub const MAX_STDERR_BYTES: usize = 1024;
pub struct LineReader { limit: usize }
pub struct FrameCodec { codec: u8 }
pub fn read_line(input: &str) -> usize { 0 }
pub fn decode_frame(bytes: &[u8]) -> usize { 0 }
"""


def _corpus(tmp_path: Path) -> Path:
    """搭一个有真实标识符的小语料。"""

    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "lib.rs").write_text(CORPUS, encoding="utf-8")
    return tmp_path


def _report(tmp_path: Path, sections: dict[str, str], name: str = "report.md") -> Path:
    """写一份报告；sections 是"小节名 → 正文"。"""

    text = "".join(f"# {title}\n{body}\n\n" for title, body in sections.items())
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path / name


def _full_body(idents: Sequence[str], paths: str = "src/lib.rs") -> str:
    """一段同时回答 5 类问题、且每类都引用了代码级证据的正文。"""

    function, type_name, constant = idents
    return (
        f"职责：负责解析，入口是 `{function}`。\n\n"
        f"交互：被 `{paths}` 里的调用方使用。\n\n"
        f"关键数据结构：核心类型 `{type_name}`。\n\n"
        f"关键参数：上限常量 `{constant}`。\n\n"
        f"失败模式：出错时 `{function}` 报错并降级。\n"
    )


def test_identifiers_per_section_passes_on_real_identifiers(tmp_path: Path) -> None:
    """正例：每节引用真实存在的常量/类型/函数 → PASS。"""

    workspace = _corpus(tmp_path)
    _report(
        workspace,
        {
            "protocol": _full_body(["decode_frame", "FrameCodec", "MAX_STDOUT_BYTES"]),
            "config": _full_body(["read_line", "LineReader", "MAX_STDERR_BYTES"]),
        },
    )

    results = run_checks(
        [
            {
                "kind": "identifiers_per_section",
                "path": "report.md",
                "sections": ["protocol", "config"],
                "min_identifiers": 3,
            }
        ],
        workspace,
    )

    assert results[0].passed, results[0].detail


def test_identifiers_per_section_fails_on_fabricated_identifiers(tmp_path: Path) -> None:
    """反例：编造标识符 → FAIL（存在性核验是这一版的命门）。"""

    workspace = _corpus(tmp_path)
    _report(
        workspace,
        {
            "protocol": _full_body(["FAKE_FN", "FakeType", "FAKE_CONST"]),
            "config": _full_body(["made_up", "GhostStruct", "NOT_REAL_CONST"]),
        },
    )

    results = run_checks(
        [
            {
                "kind": "identifiers_per_section",
                "path": "report.md",
                "sections": ["protocol", "config"],
                "min_identifiers": 3,
            }
        ],
        workspace,
    )

    assert not results[0].passed
    assert "不足" in results[0].detail


def test_identifier_deliverable_and_written_files_are_excluded(tmp_path: Path) -> None:
    """交付物与本次被写过的文件不算"语料"：自己写的名字不能自动算存在。"""

    workspace = _corpus(tmp_path)
    # 报告里内联定义了一个标识符，又引用它——如果不排除交付物，这就是自证
    (workspace / "report.md").write_text(
        "# protocol\n`INVENTED_HERE` 与 `src/lib.rs`\n", encoding="utf-8"
    )
    index = CorpusIndex(workspace, exclude=["report.md"])

    assert index.contains("MAX_STDOUT_BYTES")
    assert not index.contains("INVENTED_HERE")


def test_identifiers_per_section_dedups_across_sections(tmp_path: Path) -> None:
    """同一标识符跨节不重复计数：第二节全用第一节用过的 → 该节判不足。"""

    workspace = _corpus(tmp_path)
    body = "职责：`decode_frame`、`FrameCodec`、`MAX_STDOUT_BYTES` 都在 `src/lib.rs`。\n"
    _report(workspace, {"protocol": body, "config": body})

    results = run_checks(
        [
            {
                "kind": "identifiers_per_section",
                "path": "report.md",
                "sections": ["protocol", "config"],
                "min_identifiers": 3,
            }
        ],
        workspace,
    )

    assert not results[0].passed
    assert "config(0/3" in results[0].detail


def test_identifiers_per_section_requires_all_categories(tmp_path: Path) -> None:
    """类别不全（只有常量、没有类型/函数）→ FAIL，防止"抄一堆常量"刷分。"""

    workspace = _corpus(tmp_path)
    body = "职责：`MAX_STDOUT_BYTES`、`MAX_STDERR_BYTES` 与 `src/lib.rs`。\n"
    _report(workspace, {"protocol": body, "config": body.replace("protocol", "config")})

    results = run_checks(
        [
            {
                "kind": "identifiers_per_section",
                "path": "report.md",
                "sections": ["protocol", "config"],
                "min_identifiers": 2,
            }
        ],
        workspace,
    )

    assert not results[0].passed
    assert "缺类别" in results[0].detail


def test_identifiers_per_section_is_bounded_and_marks_unverified(tmp_path: Path) -> None:
    """有界性：核验次数超上限 → unverified（既不算通过、也不算普通失败）。"""

    workspace = _corpus(tmp_path)
    many = " ".join(f"`ident_{i}`" for i in range(10))
    _report(workspace, {"protocol": f"职责：{many} 与 `src/lib.rs`。\n"})

    results = run_checks(
        [
            {
                "kind": "identifiers_per_section",
                "path": "report.md",
                "sections": ["protocol"],
                "max_lookups": 3,
            }
        ],
        workspace,
    )

    assert results[0].unverified
    assert not results[0].passed
    assert results[0].status == "unverified"
    assert "上限" in results[0].detail


def test_question_types_per_section_requires_all_five(tmp_path: Path) -> None:
    """5 类问题缺任一 → FAIL；补全后 PASS。"""

    workspace = _corpus(tmp_path)
    complete = _full_body(["decode_frame", "FrameCodec", "MAX_STDOUT_BYTES"])
    # 整段删掉"失败模式"（只改标题没用：那段的"报错/降级"仍会命中关键词）
    without_failure = "\n\n".join(
        part for part in complete.split("\n\n") if "失败模式" not in part
    )
    _report(workspace, {"protocol": without_failure})

    results = run_checks(
        [{"kind": "question_types_per_section", "path": "report.md", "sections": ["protocol"]}],
        workspace,
    )
    assert not results[0].passed
    assert "失败模式" in results[0].detail

    _report(
        workspace,
        {"protocol": _full_body(["decode_frame", "FrameCodec", "MAX_STDOUT_BYTES"])},
    )
    results = run_checks(
        [{"kind": "question_types_per_section", "path": "report.md", "sections": ["protocol"]}],
        workspace,
    )
    assert results[0].passed, results[0].detail


def test_question_types_need_code_citation_not_just_keywords(tmp_path: Path) -> None:
    """只堆关键词、不引代码证据 → 不算回答了该类问题。"""

    workspace = _corpus(tmp_path)
    _report(
        workspace,
        {"protocol": "职责：负责解析。\n\n交互：与别人交互。\n\n关键数据结构：有数据结构。\n\n关键参数：有参数。\n\n失败模式：会失败。\n"},
    )

    results = run_checks(
        [{"kind": "question_types_per_section", "path": "report.md", "sections": ["protocol"]}],
        workspace,
    )

    assert not results[0].passed
    assert len(QUESTION_TYPES) == 5


def test_validate_checks_rejects_unknown_kind_and_missing_params() -> None:
    """设置期校验：未知检查器 / 缺参数 → 直接报错，不许静默跳过。"""

    with pytest.raises(ValueError, match="不支持的检查器"):
        validate_checks([{"kind": "no_such_checker"}])
    with pytest.raises(ValueError, match="缺少必填参数"):
        validate_checks([{"kind": "sections_cover", "path": "report.md"}])
    with pytest.raises(ValueError, match="必须是列表"):
        validate_checks([{"kind": "sections_cover", "path": "report.md", "sections": "protocol"}])
    validate_checks(subsystem_checks(["protocol", "config"], "report.md"))


def test_templated_criteria_leaks_no_hidden_checklist_items() -> None:
    """模板生成的 criteria 只规定"问哪几类问题"，不得泄露任何隐藏清单事实。"""

    from evaluation.big_repo_read_score import SUBSYSTEM_CHECKLIST
    from evaluation.read_summary_score import CHECKLIST

    text = subsystem_criteria_text(["protocol", "config", "mcp"], "report.md")
    hidden = [item.key for item in CHECKLIST] + [
        item.key for group in SUBSYSTEM_CHECKLIST.values() for item in group
    ]

    assert hidden
    for key in hidden:
        assert key not in text


def test_template_covers_all_of_its_own_criteria_lines() -> None:
    """模板生成的 criteria 每一行都必须被某条脚本检查覆盖。

    否则那一行会被当成"无脚本证据"而只能判 inconclusive——
    等于把**已经能脚本证明**的条目误判成"没法验证"（真机踩过：
    报告 5 项检查全过，却因为「职责/交互/…」五行被判 inconclusive 而无法通过）。
    """

    from core.completion_evidence import CheckResult, uncovered_criteria

    specs = subsystem_checks(["protocol", "config", "mcp"], "report.md")
    criteria = subsystem_criteria_text(["protocol", "config", "mcp"], "report.md")
    results = tuple(CheckResult(str(spec["kind"]), True, "") for spec in specs)

    assert uncovered_criteria(criteria, specs, results) == ()


def test_unverified_check_criterion_falls_back_to_uncovered() -> None:
    """检查没核验完（unverified）时，它覆盖的条目必须重新变成"无证据"。"""

    from core.completion_evidence import CheckResult, uncovered_criteria

    specs = [{"kind": "identifiers_per_section", "covers": "真实存在的标识符"}]
    criteria = "每节必须写出真实存在的标识符"
    verified = (CheckResult("identifiers_per_section", True, ""),)
    unverified = (CheckResult("identifiers_per_section", False, "", unverified=True),)

    assert uncovered_criteria(criteria, specs, verified) == ()
    assert uncovered_criteria(criteria, specs, unverified) == ("每节必须写出真实存在的标识符",)
