"""中等语料上的完成门实验（codex 的 protocol + config + mcp 子集）。

**为什么需要这一步**：codex 全仓 83 万行时，"验证"和"探索"是同一个量级，
门永远测不出价值。这里把语料缩到**验证可行**的规模（三个子系统、约 6.8 万行），
才能回答：

> 当验证真的能判定时，门能不能把覆盖率从 X% 推到接近 100%？代价是多少 token？

子集实验里点名这三个子系统是允许的（它们本来就是任务给定的阅读范围）；
**全仓实验仍然只用泛化 criteria**，绝不点名。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from .big_repo_read_compare import EXCLUDED_DIRS, MAX_FILE_BYTES, repository_hash
from .big_repo_read_score import coverage_for_subsystems
from core.completion_evidence import subsystem_checks, subsystem_criteria_text
from .read_summary_compare import is_test_code, run as _run

SOURCE = Path("/home/kevinbarrus/projects/codex")
# 语料子集按**真实 crate** 取；隐藏清单的覆盖度按**清单分组**统计（两者名字不同）
SUBSET_CRATES = ("protocol", "config", "rmcp-client", "codex-mcp")
SUBSET_GROUPS = ("protocol", "config", "mcp")
SUBSET_DIR = Path("codex-rs")
TOKEN_FUSE = 20_000_000
TIME_FUSE_SECONDS = 3_600
REPORT_NAME = "report.md"
ARMS = ("single", "fanout_fresh")

OBJECTIVE = "阅读工作区里的三个子系统（protocol / config / mcp），产出实现剖析报告"
TASK = (
    f"你的工作区里有三个子系统的源码（{SUBSET_DIR}/{'、'.join(SUBSET_CRATES)}）。"
    f"请为**每个**子系统各写一段实现剖析，汇总到工作区根目录的 {REPORT_NAME}。"
    "每段要写清：机制、关键参数或阈值、关键数据结构，以及真实代码位置（存在的文件路径）。"
)
DELEGATION_SENTENCE = (
    " 请把三个子系统的阅读工作拆给若干只读子 Agent 并行完成，再由你汇总成报告。"
)
# v3：criteria 不列"要哪些事实"，只规定"每个子系统必须回答哪几类问题"——
# 这样既不泄露隐藏清单，又逼模型真去读代码（要举出真实标识符只能读）。
MIN_IDENTIFIERS = 3
MIN_PATHS = 2
ACCEPTANCE_CRITERIA = subsystem_criteria_text(
    SUBSET_GROUPS, REPORT_NAME, min_identifiers=MIN_IDENTIFIERS, min_paths=MIN_PATHS
)
ACCEPTANCE_CHECKS: tuple[dict[str, object], ...] = subsystem_checks(
    SUBSET_GROUPS, REPORT_NAME, min_identifiers=MIN_IDENTIFIERS, min_paths=MIN_PATHS
)
_ARM_SENTENCE = {"fanout_fresh": DELEGATION_SENTENCE}
DEFAULT_SCOUT_MODE: dict[str, str | None] = {
    "single": None,
    "fanout_fresh": "fresh",
}


def subset_scope(relative: Path) -> bool:
    """只保留三个子系统的生产 Rust 源码。"""

    parts = relative.parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return False
    if len(parts) < 3 or parts[0] != SUBSET_DIR.name or parts[1] not in SUBSET_CRATES:
        return False
    return relative.suffix == ".rs" and not is_test_code(relative)


def prepare(arm: str, source: Path = SOURCE) -> Path:
    """只复制三个子系统的生产源码，形成一个"验证可行"的中等语料。"""

    root = Path(tempfile.mkdtemp(prefix=f"epsilon-mediumgate-{arm}-"))
    workspace = root / "workspace"
    files = 0
    total_bytes = 0
    for name in SUBSET_CRATES:
        crate_root = source / SUBSET_DIR / name
        for path in sorted(crate_root.rglob("*.rs")):
            relative = path.relative_to(source)
            if not subset_scope(relative):
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files += 1
            total_bytes += path.stat().st_size
    (root / "baseline.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "source": str(source),
                "subset": list(SUBSET_CRATES),
                "source_hash": repository_hash(source, MAX_FILE_BYTES, EXCLUDED_DIRS),
                "files": files,
                "bytes": total_bytes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return workspace


def subset_score(text: str, workspace: Path) -> dict[str, object]:
    """只统计三个子系统的隐藏清单覆盖率，外加路径精确率。"""

    from .read_summary_score import precision

    return {
        "coverage_source": coverage_for_subsystems(text, SUBSET_GROUPS),
        "precision": precision(text, workspace),
    }


async def run(
    workspace: Path,
    arm: str,
    scout_mode: str | None = None,
    *,
    gate_enabled: bool = True,
) -> dict[str, object]:
    """跑一档；`gate_enabled=False` 用于取"门前 X%"基线。"""

    return await _run(
        workspace,
        arm,
        scout_mode or DEFAULT_SCOUT_MODE[arm],
        source=SOURCE,
        objective=OBJECTIVE,
        task_text=TASK,
        arm_sentence=_ARM_SENTENCE,
        criteria=ACCEPTANCE_CRITERIA,
        acceptance_checks=ACCEPTANCE_CHECKS,
        score_fn=subset_score,
        token_fuse=TOKEN_FUSE,
        time_fuse_seconds=TIME_FUSE_SECONDS,
        report_name=REPORT_NAME,
        source_hash_fn=lambda root: repository_hash(root, MAX_FILE_BYTES, EXCLUDED_DIRS),
        gate_enabled=gate_enabled,
    )


def main() -> int:
    """准备副本不花钱；真机运行必须显式 --confirm。"""

    parser = argparse.ArgumentParser(description="中等语料上的完成门实验")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--no-gate", action="store_true", help="关闭完成门，用于取基线")
    args = parser.parse_args()

    if args.prepare:
        if not args.arm:
            parser.error("--prepare 需要 --arm")
        print(json.dumps({"workspace": str(prepare(args.arm))}, ensure_ascii=False))
        return 0
    if not args.confirm or args.workspace is None or not args.arm:
        parser.error("真机运行需要 --arm、--workspace 与 --confirm")
    workspace = args.workspace.resolve()
    if not workspace.is_dir() or not workspace.parent.name.startswith("epsilon-mediumgate-"):
        parser.error("workspace 必须是本脚本创建的独立副本")
    result = asyncio.run(run(workspace, args.arm, gate_enabled=not args.no_gate))
    output = workspace.parent / "result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    quality = result["quality"] or {}
    print(
        json.dumps(
            {
                "result": str(output),
                "tokens": result["actual_tokens_received"],
                "stop_reason": result["stop_reason"],
                "coverage": quality.get("coverage_source", {}).get("rate"),
                "verified": (result.get("completion_gate") or {}).get("verified"),
                "rejections": (result.get("completion_gate") or {}).get("rejections"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
