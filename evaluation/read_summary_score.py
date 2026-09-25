"""只读汇总实验的质量评分：12 项清单覆盖率 + 报告路径精确率。

清单在实验前就按 oncall 的 `MISSION.md` + `README.md` 定死，跑完只用脚本复算，
不做人工调参。两个分数：

- **覆盖率**：12 项中命中的项数 ÷ 12；
- **精确率**：报告里提到的模块/文件路径中真实存在的比例（防幻觉）。

关键词匹配只是代理指标，不是语义理解——结论里必须标注这一点。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# 报告里可能是"路径/文件名"的 token
_PATH_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|ts|tsx|vue|md|json|sql|ya?ml)")
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


def coverage(text: str) -> dict[str, object]:
    """计算 12 项清单的覆盖率。"""

    normalized = normalize(text)
    matched: list[str] = []
    missing: list[str] = []
    for item in CHECKLIST:
        hits = sum(1 for keyword in item.keywords if keyword in normalized)
        (matched if hits >= item.required_hits else missing).append(item.key)
    return {
        "matched": matched,
        "missing": missing,
        "matched_count": len(matched),
        "total": len(CHECKLIST),
        "rate": round(len(matched) / len(CHECKLIST), 4),
    }


def mentioned_paths(text: str) -> list[str]:
    """提取报告里提到的路径/文件名 token（去重、保序）。"""

    seen: dict[str, None] = {}
    for match in _PATH_PATTERN.finditer(text):
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


def precision(text: str, workspace: Path) -> dict[str, object]:
    """计算报告里提到的路径是否真实存在（防幻觉）。"""

    relative, names = _existing_paths(workspace)
    mentioned = mentioned_paths(text)
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
    """返回报告的质量评分（覆盖率 + 精确率）。"""

    return {
        "coverage": coverage(text),
        "precision": precision(text, workspace),
    }
