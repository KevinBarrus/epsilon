# 指令：只读汇总任务的多 Agent 对比实验（oncall）

> 本实验是本项目第一次测"**多 Agent 应该赢的场景**"（fan-out 读）。
> 前两个实验（重构任务）测的是 fan-out 写——那是多 Agent 的已知劣势场景。

## Phase 0：先提交积压（**必须先做**）

上两轮 Loop Guard v2 的代码与文档**尚未提交**，工作区有未跟踪文件。先提两个提交，再开始本实验。

**提交 1（代码）**：`src/core/loop_guard.py`、`src/core/agent_loop.py`、`src/core/subagent.py`、
`evaluation/events.py`、`evaluation/big_task_single_goal.py`、`tests/test_loop_guard.py`、
`tests/test_evaluation_events.py`、新增 `evaluation/replay_loop_guard.py`、
`tests/test_replay_loop_guard.py`、`todo/loop-guard-v2.md`

```
feat(agent): Loop Guard v2 与可回放观测，并完成真机验证

- 观测：ToolExecution/ToolCallStarted/ToolCallCancelled/ToolBatch 事件新增
  agent_run_id，AgentLoop 新增 run_id 并把父级 spawn 调用标识透传给子 Agent
- 归档：新增 evaluation/events.py:child_event_record，子事件按轮落盘
  （round / call_id / args_digest / args_preview / output_digest / error_family）
- 检测 A 增强：新增 abab_action_cycle 与 same_error_family
- 检测 B 重设计：由"没有成功的非读调用"改为"没有新事实"
- 提醒节制：每轮最多一条，按 进展不变 > 错误族 > 重复调用 > abab 取优先级
- 新增 evaluation/replay_loop_guard.py：按 run_id 回放归档
- 测试：全量 795 项通过
```

**提交 2（文档）**：更新 `problems/multi-agent-arms.md`

```
docs(problems): 补充 worktree v2 档与 Loop Guard v2 真机结论

- 增加 worktree v2 档（120.1M / 2:14:06 / 73/75 / tsc 不通过）
- "并行能否压缩墙钟"更新为明确结论：失控未复现的情况下仍是 2.83× 墙钟 /
  3.60× token、完成度更低，多 Agent 在本任务上不划算
- 记录 Loop Guard v2 真机表现：全轮注入 3 次、回放与真机一致、反事实
  v1≈304 次 → v2 3 次、288 轮合法探索零误报；纠偏的救援效果未获证据
```

## 一、实验目的与假设

**目的**：验证"**多 Agent 在只读、可扇出的任务上是否比单 Agent 更好**"。

**为什么选读任务**：读是**唯一能真正分摊**的成本——各 Scout 读**互不重叠**的部分，
不像写那样每个 Agent 都要重建上下文、还要对齐接口。业界（Claude Code / codex /
pi）也是把 subagent 用在并行只读探索上。

**假设**：
- **墙钟**：多 Agent **明显下降**；
- **token**：多 Agent **大致持平或略升**（扇出读不重复，但每个子 Agent 仍有固定开销）；
- **质量**：多 Agent **不劣于**单 Agent（分区读不会丢信息）。

若假设不成立（墙钟没降、质量更差），如实记录——那也是结论。

## 二、实验设计（三档，A/B 必做，C 选做）

**任务目标**：阅读副本里的 oncall 项目，产出 `report.md` 总结报告。

**范围**：`docs/`、`openspec/`、`apps/backend`、`apps/frontend`、顶层 `*.md`。
**排除**：`.git`、`node_modules`、`.venv`、以及任何超过 1MB 的文件。
（oncall 排除后约 3800 个文件、约 4.7 万行 py/ts。）

| 档 | 委派工具 | 提示词 |
|---|---|---|
| **A 单 Agent** | 不注册委派工具 | 中性任务描述 |
| **B 多 Agent·主动** | 注册 `spawn_agent`（只读 Scout） | 中性任务 + "请把阅读工作按主题/目录拆给若干**只读**子 Agent 并行完成，再由你汇总成报告" |
| **C 多 Agent·中性**（选做） | 注册 `spawn_agent` | **只给**中性任务描述（**不提委派**），测"模型会不会自发并行" |

**三档用完全相同的内容副本**（分别复制，内容一致）；模型、thinking 等级、任务描述
（除上述差异）全部一致。

## 三、质量评分（**必须预注册，跑之前就定死**）

清单从 oncall 的 `MISSION.md` + `README.md` 提炼，**共 12 项**：

1. 项目定位：本地优先的 AIOps 工作台
2. 技术栈：Vue 3 / FastAPI / SQLite / Milvus / LangChain / Qwen(OpenAI-compatible) / 腾讯云 CLS MCP
3. 诊断链路四个角色：**Planner / Executor / Replanner / Report**
4. 使用 **LangGraph** 编排（并说明理由）
5. 隔离模型：**单用户即单租户**（tenant 范围 = owner 用户）
6. RAG 混合召回：Milvus 向量 + 内存 **BM25L**，**RRF（k=60）**融合，再 **rerank**
7. 文档索引：后台任务 + 状态机（排队/执行中/成功/失败/取消）+ 重试/重建
8. 会话记忆模式：每 30 轮压缩 / 上下文 70% 自动压缩 / 手动压缩
9. Skill 渐进式加载：初始只注入 `name`+`description`，需要时 `load_skill`
10. 权限：密码用 **Argon2** 哈希；越权返回统一权限错误
11. 引用可解释性：同时展示向量排名+相似度、BM25 排名+分数、RRF 分数、rerank 排名+分数
12. 已知局限/未落地（`MISSION.md` 的 Out of scope：生产级自动修复、分布式恢复、通用 Tool Registry）

**两个分数**：
- **覆盖率（recall）** = 命中项数 ÷ 12；
- **精确率（precision）** = 报告里提到的**模块名/文件路径**中**真实存在**的比例（**防幻觉**，可脚本核查）。

**（可选）** 引入一个模型当裁判打分，但必须在报告里标注"主观项"。

## 四、指标（每档都要）

token（总量 + 按角色）、墙钟、模型请求数、工具错误数/错误率、委派次数、
质量（覆盖率 + 精确率）、**重复读率**（多个 Scout 是否读了同一个文件——衡量分区质量）。

## 五、实现要求

新增 `evaluation/read_summary_compare.py`：

- `--prepare`：把 oncall 复制到临时目录（按第二节的排除规则），写 `baseline.json`；
- `--arm single|multi_delegated|multi_neutral`；
- `--confirm` 才真机跑（未加时只做准备与校验）；
- 熔断：**40M token / 60 分钟墙钟**（每档独立计时；宽松但必须有界）；
- 产出：`result.json`（指标）+ `report.md`（Agent 产物）+ 轨迹归档到
  `evaluation-results/oncall-read-<arm>-<date>/`（该目录已被忽略，不入库）；
- **复用现有件**：`usage_breakdown`、`child_event_record`、`config_from_settings`（Loop Guard 接线）、
  `evaluate` 侧的 `--delegate` 工具注册路径；不要另起一套；
- **原 oncall 仓库零改动**（只改副本），并在 `result.json` 里记 `original_code_unchanged`。

## 六、验收

1. 三档（或两档）各自跑完并产出 `report.md` + `result.json`；
2. 覆盖率/精确率可脚本复算（评分脚本随实验提交在 `evaluation/` 下）；
3. 墙钟与 token 对比表可直接放进 `problems/` 文档；
4. 如实标注：**单次运行、描述性结论，不作统计推断**；
5. 若某一档撞熔断，如实记录并说明未完成。

## 七、边界

- **不改 oncall 仓库**；
- 不给子 Agent 写权限（Scout 只读）；
- 不为"让多 Agent 赢"而调提示词——两档的任务描述除委派句外**必须一致**；
- 不设轮次上限（沿用"长任务不掐死"原则），只用熔断兜底。

---

## 附：成本提示

读任务预计每档 **2~10M token**（远低于重构任务）。熔断上限 40M token / 60 分钟。
执行方在 `--confirm` 前报一次预估，用户确认后开跑。
