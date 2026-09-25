"""只读汇总实验的质量评分：文档级清单 + 源码级清单 + 路径精确率。

两张清单都在实验前按 oncall 的文档与源码定死，跑完只用脚本复算：

- **文档级覆盖率**（`CHECKLIST`，12 项）：读全文档的能力；
- **源码级覆盖率**（`CHECKLIST_SOURCE`，6 项，主指标）：答案只在源码里的问题；
- **精确率**：报告里提到的模块/文件路径中真实存在的比例（防幻觉）。

出题纪律：`CHECKLIST_SOURCE` 的每个关键词都验证过**不在** `README.md` /
`MISSION.md` / `WORK_DONE.md` / `CLAIM.md` / `docs/` / `openspec/` 里出现，
否则它就是文档题而不是源码题。

关键词匹配只是代理指标，不是语义理解——结论里必须标注这一点。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# 报告里可能是"路径/文件名"的 token
_PATH_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|ts|tsx|vue|md|json|sql|ya?ml)")
DEFAULT_PATH_PATTERN = _PATH_PATTERN
# 路径判定用的后缀集合，避免把每个文件都装进内存后再做 O(n) 扫描
_CODE_SUFFIXES = (".py", ".ts", ".tsx", ".vue", ".md", ".json", ".sql", ".yml", ".yaml")


@dataclass(frozen=True)
class ChecklistItem:
    """一项预注册的质量清单。"""

    key: str
    title: str
    keywords: tuple[str, ...]
    required_hits: int = 1


# 预注册清单：12 项，来源 MISSION.md + README.md
CHECKLIST: tuple[ChecklistItem, ...] = (
    ChecklistItem("positioning", "项目定位：本地优先的 AIOps 工作台", ("本地优先", "aiops"), 2),
    ChecklistItem(
        "stack",
        "技术栈：Vue3 / FastAPI / SQLite / Milvus / LangChain / Qwen / CLS MCP",
        ("vue", "fastapi", "sqlite", "milvus", "langchain", "qwen", "cls"),
        4,
    ),
    ChecklistItem(
        "roles",
        "诊断链路四角色：Planner / Executor / Replanner / Report",
        ("planner", "executor", "replanner", "report"),
        3,
    ),
    ChecklistItem("langgraph", "用 LangGraph 编排", ("langgraph",), 1),
    ChecklistItem(
        "tenancy",
        "隔离模型：单用户即单租户",
        ("单用户即单租户", "单租户", "tenant"),
        1,
    ),
    ChecklistItem(
        "hybrid_recall",
        "混合召回：Milvus 向量 + BM25L + RRF(k=60) + rerank",
        ("bm25", "rrf", "rerank"),
        2,
    ),
    ChecklistItem(
        "index_jobs",
        "文档索引：后台任务 + 状态机（排队/执行中/成功/失败/取消）+ 重试",
        ("后台任务", "状态机", "重试", "排队", "索引任务"),
        2,
    ),
    ChecklistItem(
        "memory",
        "会话记忆模式：每 30 轮压缩 / 上下文 70% 自动压缩 / 手动压缩",
        ("30 轮", "70%", "压缩"),
        2,
    ),
    ChecklistItem(
        "skill",
        "Skill 渐进式加载：初始只注入 name+description，需要时 load_skill",
        ("渐进", "load_skill", "description"),
        2,
    ),
    ChecklistItem("argon2", "权限：密码 Argon2 哈希；越权统一权限错误", ("argon2",), 1),
    ChecklistItem(
        "explainability",
        "引用可解释性：向量排名/相似度 + BM25 排名/分数 + RRF 分数 + rerank 排名",
        ("向量排名", "bm25 排名", "rrf 分数", "rerank 排名", "相似度"),
        2,
    ),
    ChecklistItem(
        "limitations",
        "已知局限 / 未落地（生产级自动修复、分布式恢复、通用 Tool Registry）",
        ("局限", "未落地", "out of scope", "自动修复"),
        1,
    ),
)


def normalize(text: str) -> str:
    """把报告文本归一化成便于关键词匹配的形式。"""

    return re.sub(r"\s+", " ", text).lower()


# 源码级清单（6 项，主指标）：关键词全部验证过只存在于源码里
CHECKLIST_SOURCE: tuple[ChecklistItem, ...] = (
    ChecklistItem(
        "source_tokenize",
        "BM25 那一路的中文分词：正则 token 规则 + 中文片段单独切",
        ("_token_pattern", "_is_chinese_segment", "tokenize_hybrid_text"),
        1,
    ),
    ChecklistItem(
        "source_belief_threshold",
        "信念压缩阈值：≥3 次观测且成功概率 ≥ 0.72",
        ("0.72",),
        1,
    ),
    ChecklistItem(
        "source_memory_modes",
        "会话记忆三档的字面枚举值",
        ("every_30_turns", "context_70_percent"),
        1,
    ),
    ChecklistItem(
        "source_roles_layout",
        "四角色在同一文件内（LangGraph 节点装配）而非四个文件",
        ("add_node", "stategraph", "plan_origin"),
        1,
    ),
    ChecklistItem(
        "source_evidence_table",
        "compressed_tool_evidence 表：工具证据压缩后落库",
        (
            "compressed_tool_evidence",
            "add_compressed_tool_evidence",
            "_compressed_tool_evidence_record",
            "202607110012",
        ),
        # 题面会提到表名，因此要求 2 项命中：至少还要说出现源码才有的写入函数 / 迁移名
        2,
    ),
    ChecklistItem(
        "source_job_states",
        "索引任务状态机的字面值",
        ("queued", "running", "succeeded", "failed", "cancelled"),
        3,
    ),
)


def coverage(text: str) -> dict[str, object]:
    """计算文档级清单（12 项）的覆盖率。"""

    return _coverage(text, CHECKLIST)


def coverage_source(text: str) -> dict[str, object]:
    """计算源码级清单（6 项）的覆盖率。"""

    return _coverage(text, CHECKLIST_SOURCE)


def _coverage(text: str, checklist: tuple[ChecklistItem, ...]) -> dict[str, object]:
    """按给定清单计算覆盖率。"""

    normalized = normalize(text)
    matched: list[str] = []
    missing: list[str] = []
    for item in checklist:
        hits = sum(1 for keyword in item.keywords if keyword in normalized)
        (matched if hits >= item.required_hits else missing).append(item.key)
    return {
        "matched": matched,
        "missing": missing,
        "matched_count": len(matched),
        "total": len(checklist),
        "rate": round(len(matched) / len(checklist), 4),
    }


def mentioned_paths(text: str, pattern: re.Pattern[str] = DEFAULT_PATH_PATTERN) -> list[str]:
    """提取报告里提到的路径/文件名 token（去重、保序）。"""

    seen: dict[str, None] = {}
    for match in pattern.finditer(text):
        token = match.group(0).strip("./")
        if token and token not in seen:
            seen[token] = None
    return list(seen)


def _existing_paths(workspace: Path) -> tuple[set[str], set[str]]:
    """返回工作区相对路径集合与文件名集合。"""

    relative: set[str] = set()
    names: set[str] = set()
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace).as_posix()
        relative.add(rel)
        names.add(path.name)
    return relative, names


def precision(
    text: str,
    workspace: Path,
    pattern: re.Pattern[str] = DEFAULT_PATH_PATTERN,
) -> dict[str, object]:
    """计算报告里提到的路径是否真实存在（防幻觉）。"""

    relative, names = _existing_paths(workspace)
    mentioned = mentioned_paths(text, pattern)
    existing: list[str] = []
    hallucinated: list[str] = []
    for token in mentioned:
        stripped = token.strip("./")
        is_real = (
            stripped in relative
            or stripped in names
            or any(path.endswith("/" + stripped) for path in relative)
        )
        (existing if is_real else hallucinated).append(token)
    total = len(mentioned)
    return {
        "mentioned": total,
        "existing": len(existing),
        "hallucinated": hallucinated[:20],
        "rate": round(len(existing) / total, 4) if total else None,
    }


def score(text: str, workspace: Path) -> dict[str, object]:
    """返回报告的质量评分（文档级覆盖率 + 源码级覆盖率 + 精确率）。"""

    return {
        "coverage": coverage(text),
        "coverage_source": coverage_source(text),
        "precision": precision(text, workspace),
    }
