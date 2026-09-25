"""只读汇总实验的质量评分测试。"""

from pathlib import Path

from evaluation.read_summary_score import (
    CHECKLIST,
    coverage,
    mentioned_paths,
    precision,
    score,
)

FULL_REPORT = """
# oncall 项目总结

oncall 是一个本地优先的 AIOps 工作台，前端 Vue 3、后端 FastAPI，SQLite 保存数据，
Milvus 保存向量，用 LangChain 与 Qwen（OpenAI-compatible）调用模型，通过腾讯云 CLS MCP 访问日志。

诊断链路是 Planner -> Executor -> Replanner -> Report，用 LangGraph 编排。
隔离模型是单用户即单租户，tenant 范围等于 owner 用户。

检索侧：Milvus 向量召回与内存 BM25 并行，用 RRF 融合候选，再 rerank 精排。
文档索引走后台任务与状态机（排队/执行中/成功/失败/取消），支持重试。
会话记忆支持每 30 轮压缩、上下文 70% 自动压缩与手动压缩。
Skill 渐进式加载，初始只注入 name 和 description，需要时 load_skill。
密码用 Argon2 哈希；引用同时展示向量排名、BM25 排名、RRF 分数与 rerank 排名。

已知局限：生产级自动修复、分布式恢复留作 out of scope，属于未落地能力。
实现见 apps/backend/src/oncall/main.py 与 docs/index.md。
"""


def test_checklist_has_twelve_items() -> None:
    """清单必须固定为 12 项（预注册，跑之前定死）。"""

    assert len(CHECKLIST) == 12


def test_full_report_matches_all_items() -> None:
    """覆盖全部要点的报告应拿满覆盖率。"""

    result = coverage(FULL_REPORT)
    assert result["matched_count"] == 12
    assert result["rate"] == 1.0
    assert result["missing"] == []


def test_empty_report_matches_nothing() -> None:
    """空报告覆盖率为 0。"""

    result = coverage("")
    assert result["matched_count"] == 0
    assert result["rate"] == 0.0
    assert len(result["missing"]) == 12


def test_partial_report_reports_missing_items() -> None:
    """只写了部分要点时，缺失项要被点名。"""

    result = coverage("这是一个本地优先的 AIOps 工作台，用 LangGraph 编排。")
    assert "langgraph" in result["matched"]
    assert "argon2" in result["missing"]
    assert 0 < result["rate"] < 1


def test_mentioned_paths_extracts_and_dedupes() -> None:
    """路径提取会去掉重复并保留出现顺序。"""

    text = "见 apps/backend/src/oncall/main.py，也见 apps/backend/src/oncall/main.py 和 docs/index.md。"
    assert mentioned_paths(text) == ["apps/backend/src/oncall/main.py", "docs/index.md"]


def test_precision_separates_real_and_hallucinated_paths(tmp_path: Path) -> None:
    """真实存在的路径计入精确率，编造的路径被点名。"""

    real = tmp_path / "apps/backend/src/oncall/main.py"
    real.parent.mkdir(parents=True)
    real.write_text("x", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/index.md").write_text("y", encoding="utf-8")

    text = "见 apps/backend/src/oncall/main.py、src/oncall/main.py、docs/index.md 与 src/oncall/ghost.py"
    result = precision(text, tmp_path)

    assert result["existing"] == 3
    assert result["hallucinated"] == ["src/oncall/ghost.py"]
    assert result["rate"] == 0.75


def test_precision_accepts_basename_only_mention(tmp_path: Path) -> None:
    """只写文件名时按工作区里的同名文件判定。"""

    (tmp_path / "MISSION.md").write_text("mission", encoding="utf-8")
    result = precision("见 MISSION.md", tmp_path)

    assert result["rate"] == 1.0


def test_score_combines_both_metrics(tmp_path: Path) -> None:
    """score 同时返回覆盖率与精确率。"""

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/index.md").write_text("x", encoding="utf-8")
    result = score("本地优先的 AIOps 项目，见 docs/index.md", tmp_path)

    assert set(result) == {"coverage", "precision"}
    assert result["precision"]["rate"] == 1.0
