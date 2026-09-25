"""大语料只读对比实验（codex）：先容量探测，再跑 A/B/C。

语料是 codex 仓库。范围 = **生产 Rust + 非代码资源**，排除
`.git` / `target` / `node_modules`、**测试代码**（`tests/`、`test/`、`*tests.rs`、`*_test.rs`）
与单文件 > 2MB。

排除测试代码的理由：测试占 Rust 总量约 42%（57 万行），而任务是"为子系统写实现剖析"，
读测试代码对目标没有贡献。

- A `single`：单 Agent 串行读；
- B `fanout_fresh`：fresh 只读子 Agent 扇出（各读不相关子系统）；
- C `map_then_fork`：父先测绘 → fork 子 Agent 继承项目图后深读。

**Phase 1 只用 A 做容量探测**：若 A 没撞熔断且覆盖了清单大部分，说明语料还是不够大。

任务描述只给**泛化要求**，绝不列举清单内容（清单是隐藏的评分标准）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from .big_repo_read_score import score as codex_score
from .read_summary_compare import (
    copy_repository,
    is_test_code,
    repository_hash,
    run as _run,
)

SOURCE = Path("/home/kevinbarrus/projects/codex")
EXCLUDED_DIRS = {".git", "target", "node_modules"}
MAX_FILE_BYTES = 2_000_000
TOKEN_FUSE = 40_000_000
TIME_FUSE_SECONDS = 5_400  # 90 分钟
REPORT_NAME = "report.md"
ARMS = ("single", "fanout_fresh", "map_then_fork")
DEFAULT_SCOUT_MODE: dict[str, str | None] = {
    "single": None,
    "fanout_fresh": "fresh",
    "map_then_fork": "fork",
}

OBJECTIVE = "通读工作区里的 codex 仓库，产出覆盖各核心子系统的实现剖析报告"
# 只写泛化要求，不列举任何清单条目
TASK = (
    "你的工作区是 codex 仓库的副本。请通读它的源码，"
    f"为每个核心子系统写一段实现剖析，并把报告写到工作区根目录的 {REPORT_NAME}。"
    "每段剖析要说清：机制、关键参数或阈值、关键数据结构、以及对应的代码位置（真实文件路径与标识符）。"
    "报告要覆盖所有主要子系统，每个子系统至少一段；"
    "无法确定的地方明确写“未能确定”，不许编造。"
)
DELEGATION_SENTENCE = (
    " 请把阅读工作按子系统拆给若干只读子 Agent 并行完成，再由你汇总成报告。"
)
MAP_FIRST_SENTENCE = (
    " 先自己快速测绘仓库结构（主要 crate / 子系统以及它们的关系），"
    "再把测绘结果与你已读到的内容交给若干只读子 Agent，"
    "让它们在你已有上下文的基础上深读各自负责的子系统。"
)
_ARM_SENTENCE: dict[str, str] = {
    "fanout_fresh": DELEGATION_SENTENCE,
    "map_then_fork": MAP_FIRST_SENTENCE,
}


def codex_scope(relative: Path) -> bool:
    """codex 语料范围：生产 Rust + docs/ + scripts/ + 顶层 Markdown。

    排除测试 Rust（约占 42%）与生成物噪声（lock/snap/大 JSON 等），
    让覆盖率反映"读了多少实现"。
    """

    parts = relative.parts
    if relative.suffix == ".rs":
        return not is_test_code(relative)
    if parts and parts[0] in {"docs", "scripts"}:
        return True
    return len(parts) == 1 and relative.suffix.lower() in {".md", ".mdx"}


def prepare(arm: str, source: Path = SOURCE) -> Path:
    """创建 codex 副本（按范围谓词 + 2MB 阈值），并记录基线哈希。"""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    root = Path(tempfile.mkdtemp(prefix=f"epsilon-bigrepo-{arm}-"))
    workspace = root / "workspace"
    stats = copy_repository(
        source,
        workspace,
        max_file_bytes=MAX_FILE_BYTES,
        excluded_dirs=EXCLUDED_DIRS,
        skip_tests=True,
        include=codex_scope,
    )
    (root / "baseline.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "source": str(source),
                "source_hash": repository_hash(
                    source, MAX_FILE_BYTES, EXCLUDED_DIRS, skip_tests=True, include=codex_scope
                ),
                "copy_hash": repository_hash(
                    workspace, MAX_FILE_BYTES, EXCLUDED_DIRS, skip_tests=True, include=codex_scope
                ),
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return workspace


async def run(workspace: Path, arm: str, scout_mode: str | None = None) -> dict[str, object]:
    """用 codex 的 profile 跑一档（复用只读汇总实验的 harness）。"""

    resolved_mode = scout_mode or DEFAULT_SCOUT_MODE[arm]
    return await _run(
        workspace,
        arm,
        resolved_mode,
        source=SOURCE,
        objective=OBJECTIVE,
        task_text=TASK,
        arm_sentence=_ARM_SENTENCE,
        score_fn=codex_score,
        token_fuse=TOKEN_FUSE,
        time_fuse_seconds=TIME_FUSE_SECONDS,
        report_name=REPORT_NAME,
    )


def main() -> int:
    """准备副本不花钱；真机运行必须显式 --confirm。"""

    parser = argparse.ArgumentParser(description="大语料只读对比实验（codex）")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--scout-mode", choices=("fresh", "fork", "fork_last_n"))
    args = parser.parse_args()

    if args.prepare:
        if not args.arm:
            parser.error("--prepare 需要 --arm")
        workspace = prepare(args.arm, args.source)
        print(json.dumps({"workspace": str(workspace)}, ensure_ascii=False))
        return 0
    if not args.confirm or args.workspace is None or not args.arm:
        parser.error("真机运行需要 --arm、--workspace 与 --confirm")
    workspace = args.workspace.resolve()
    if not workspace.is_dir() or not workspace.parent.name.startswith("epsilon-bigrepo-"):
        parser.error("workspace 必须是本脚本创建的独立副本")
    baseline = json.loads((workspace.parent / "baseline.json").read_text(encoding="utf-8"))
    if baseline.get("arm") != args.arm:
        parser.error("运行档位必须与副本准备时的档位一致")
    result = asyncio.run(run(workspace, args.arm, args.scout_mode))
    output = workspace.parent / "result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(output),
                "stop_reason": result["stop_reason"],
                "actual_tokens_received": result["actual_tokens_received"],
                "duration_seconds": result["duration_seconds"],
                "source_coverage": result["quality"]["coverage_source"]["rate"]
                if result["quality"]
                else None,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
