# 指令：只读汇总任务的多 Agent 对比实验（oncall）

> **本次是"修正版"**：A 档已跑完，暴露了任务太浅的问题。先改任务与评分，**重跑 A**，再跑 B/C。

## 零、当前进度与本次要改什么

**A 档（单 Agent 串行读）已跑完**：

| 项 | 值 |
|---|---|
| token | 1,858,503 |
| 墙钟 | **104 秒** |
| 覆盖率 | 9/12（0.75）—— 缺 memory 三档压缩、Skill 渐进式加载、引用可解释性 |
| 精确率 | 55/57（0.965）—— 幻觉路径 `template.json`、`ragas_evaluation.py` |
| 重复读率 | 0.089（45 次读、41 份唯一） |

**两个关键判断**：

1. **任务太浅，测不出扇出读的价值**。12 项清单几乎全在 `README.md` / `MISSION.md` 里
   （`grep README.md` 命中 8 处 RRF/BM25/Argon2/渐进式 Skill/30 轮）。A 的 9/12 本质是
   "**把那份 README 读全了**"，不是"读得深"。而 oncall 的真实体量在源码——
   `apps/backend/src/super_ai` 有 **57 个 Python 文件、19,918 行**，A 只读了其中很小一部分。
   **信息集中在文档里，就没有可扇出的读。**
2. **"压缩墙钟"这个维度作废**：A 只用 **104 秒**，并行最多压到几十秒，协调开销就吃回去了。
   本实验的主战场改为**覆盖率**与**精确率**。

**本次要改四件事**：① 任务加"源码级"要求；② 评分拆成两张清单；③ 档位定稿；④ **改完重跑 A**，再跑 B/C。

## 一、实验目的与假设

**目的**：验证"**多 Agent 在只读、可扇出的任务上是否比单 Agent 更好**"。

**修正后的假设**（不再含墙钟）：

- **覆盖率（源码级）**：多 Agent **应高于**单 Agent——这是扇出读唯一可能赢的维度；
- **token**：多 Agent **更高**（子 Agent 有固定开销）；
- **精确率**：多 Agent **不应低于**单 Agent（但幻觉机会更多，必须盯）。

## 二、任务与档位

### 2.1 任务描述（在原任务后追加）

> 除项目级总结外，报告还必须回答下面的**源码级**问题（答案不在文档里，必须读源码）；
> 无法确定时明确写"未能确定"，**不许编造**。

### 2.2 档位

| 档 | arm 名 | 内容 |
|---|---|---|
| **A** | `single` | 单 Agent 串行读 |
| **B** | `fanout_fresh` | **fresh** Scout 扇出并行读（各读不相关分区，不共享上下文） |
| **C** | `map_then_fork` | **父先做项目测绘** → **fork** Scout 继承项目图后深读各分区（延续式探索） |

- `multi_neutral`（模型自发是否委派）**选做**，不计入主对照；
- **B 与 C 是本次的核心对照**：同一任务、同样扇出，唯一差别是**子 Agent 是否继承父的上下文**。

## 三、评分：两张清单（**必须预注册，跑之前定死**）

### 3.1 文档级清单（原 12 项，保留）

沿用 `evaluation/read_summary_score.py` 现有 `CHECKLIST`，作为"读全文档"的能力指标。

### 3.2 源码级清单（**新增 6 项**，本次的主指标）

| # | 问题 | 代码位置（已核实） | 判定要点 |
|---|---|---|---|
| 1 | BM25 那一路的**中文分词**怎么做？ | `retrieval/hybrid.py` | 提到正则 token 规则（`_TOKEN_PATTERN`）、`tokenize_hybrid_text`、**中文片段单独切**（`_is_chinese_segment`）；只说"用了 BM25"不算 |
| 2 | **信念压缩**的确切阈值规则？ | `aiops/sop_belief.py` | **≥3 次观测 且 成功概率 ≥ 0.72** 才压缩（`n >= 3 and p >= 0.72`） |
| 3 | 会话记忆三档的**字面枚举值**，以及压缩发生在**哪一层**？ | `api/app.py:164` | `Literal["every_30_turns", "context_70_percent", "manual"]`；并说明压缩在**存储层**还是**上下文构建层** |
| 4 | Planner / Executor / Replanner / Report 的**实现位置**与**状态传递字段**？ | `aiops/`（只有 `cases.py`/`diagnostics.py`/`fixtures.py`/`sop_belief.py`） | 指出四者**在同一 `diagnostics.py` 内**（**不是四个文件**），并说出关键状态字段 |
| 5 | `compressed_tool_evidence` 表的**用途与关键字段**？ | `memory/sqlite.py`（`_compressed_tool_evidence_record`）、迁移 `202607110012_add_compressed_tool_evidence.py` | 说出"工具证据压缩后落库" + 关键字段（如 `source_hash`） |
| 6 | 索引任务状态机的**字面枚举值**与**失败重试**路径？ | `memory/repositories.py`、`memory/sqlite.py` | 说出至少 3 个字面状态值，以及失败后如何重试/重建 |

**出题纪律**：每一项在写入 `CHECKLIST_SOURCE` 前，**必须确认答案不在 `README.md` / `MISSION.md` / `docs/` 里**
（例如 `RRF_K = 60` 就**在 README 里**，所以不能当源码题）。

**两个分数**：**文档级覆盖率**、**源码级覆盖率**（各 = 命中项 ÷ 总数）；
外加 **精确率**（报告提到的模块/路径中真实存在的比例，脚本核查，**防幻觉**）。

## 四、指标（每档都要）

token（总量 + 按角色）、**文档级覆盖率**、**源码级覆盖率**、**精确率**、
读调用次数、**重复读率**、委派次数、**缓存命中率**（fork/fresh 分开）。
墙钟**记录但不作主指标**。

## 五、执行顺序

1. **改评分脚本** `evaluation/read_summary_score.py`：新增 `CHECKLIST_SOURCE`（6 项）与
   `coverage_source()`，输出两个覆盖率；补单测（用 A 档报告验证"6 项里应命中 0~2 项"）；
2. **改任务描述** `evaluation/read_summary_compare.py`：追加"源码级问题"段（对**所有档完全相同**）；
3. **重跑 A**（`--arm single`，约 2M / 2 分钟）——任务变了必须重跑，否则不公平；
4. **跑 B**（`--arm fanout_fresh`）；
5. **跑 C**（`--arm map_then_fork`，`--scout-mode fork`）；
6. 产出对照表，写入 `problems/oncall-read-compare.md`（**并归档三档产物到 `evaluation-results/`**，
   当前 A 档产物还在 `/tmp/epsilon-read-single-*`，**属易失**，必须固化）。

## 六、验收

1. 三档都产出 `report.md` + `result.json`，且**归档到 `evaluation-results/oncall-read-<arm>-<date>/`**；
2. 两个覆盖率与精确率**可脚本复算**；
3. 对照表含四维（token / 文档级覆盖率 / 源码级覆盖率 / 精确率）+ 读调用次数与缓存命中率；
4. 如实标注：**单次运行、描述性结论，不作统计推断**；
5. **重点回答**：B/C 相对 A，**源码级覆盖率**有没有提高？提高的代价是几倍 token？
   C 相对 B，**继承父上下文**有没有让"深读"更有效？

## 七、边界

- **不改 oncall 仓库**（只改副本），`result.json` 记 `original_code_unchanged`；
- 子 Agent **只读**（Scout），不给写权限；
- 任务描述**除档位说明外必须一致**，不为"让多 Agent 赢"调提示词；
- **不设轮次上限**，只用熔断兜底：**40M token / 60 分钟每档**；
- fork 档必须用**放宽后的引用口径**（已读过可直接引用，不必重读；未出现过的不得引用）。

---

## 附：成本

每档预计 **2~10M token**（A 实测 1.86M）。**逐档跑、逐档汇报**，每档开跑前报预估。
三档合计预计 **< 30M token**。
