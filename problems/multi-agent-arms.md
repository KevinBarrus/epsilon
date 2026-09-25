# 多 Agent 三档对照：单 Agent / 共享工作区 / worktree 并行（三轮）

同一任务（把副本 `src/core` 的 75 个 Python 模块重构成等价的 TypeScript）、
同 goal、同模型 `deepseek-flash`（`thinking=high`）、同安全熔断口径。
每档只跑一次，是描述性对照，不作统计结论。

## 一、结果（三档，其中 worktree 跑了三轮）

| 档 | token | 墙钟 | 墙钟相对单 Agent | token 相对单 Agent | 工具错误 | 完成度 | tsc | 终止 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 单 Agent（有 goal） | 33,317,726 | 47:24 | 1.00× | 1.00× | 16 | **75/75** | 通过 | Goal complete |
| 多 Agent 共享工作区 | 58,546,994 | 1:56:13 | 2.45× | 1.76× | 69 | 45/75 | 独立复验通过 | 50M token 熔断 |
| worktree 并行（旧，污染） | ≥83,815,202 | ~2:00 | ~2.53× | ≥2.52× | ≥114 | 71/75 | 独立复验通过 | 墙钟到期，收尾异常 |
| worktree 并行（v1 干净） | 120,010,128 | 2:34:33 | **3.26×** | **3.60×** | 95 | 49/75 | 通过（harness） | 120M token 熔断 |
| worktree 并行（v2，带 Loop Guard） | 120,077,810 | **2:14:06** | **2.83×** | **3.60×** | 217 | 73/75 | **不通过** | 120M token 熔断 |

v2 档完成度比 v1 高（73/75 对 49/75），但 **tsc 不通过**（`core/ui.ts(536,9): error TS2353`），
按口径只能判 incomplete：“文件都存在”不等于“类型正确”。

两个比例的算法：分别用该档的墙钟/token 除以单 Agent 的 2844.5 秒 / 33,317,726 token。

## 二、三个可靠结论

1. **多 Agent 的 token 代价稳健更高**：共享档 token 为单 Agent 的 **1.76×**、worktree 档
   **3.60×**（旧轮 ≥2.52× 为下界）。原因是文本通信带来的"重读成本"——子 Agent 拿不到
   父 Agent 的上下文，必须自己再读一遍。**这个成本改不掉**，加并发、加隔离都不影响它。

2. **worktree 隔离 + 并行机制本身可用**：干净重跑里 merge 冲突 **0**（旧轮 6 次全是
   `.npm` 假冲突）、实际并发峰值 **4**、7 个 Worker 干净合并。修掉"混合批次整批串行化"后，
   Worker 确实真并发了。

3. **“并行能不能把墙钟压回来”：在失控未复现的情况下，仍然是否定的**。v2 把最大单 Worker 从
   56.2% / 97 分钟压到 11.3% / 24.8 分钟，墙钟从 3.26× 改善到 **2.83×**，但
   **仍是单 Agent 的 2.83×、token 仍是 3.60×，且完成度更低（73/75 但 tsc 不通过，对 75/75 通过）**。
   原因：墙钟主要由“总工作量 × 重读成本”决定，而重读成本随 Agent 数单调上升，
   并行只能把它们重叠，压不过重读带来的额外总量。

## 三、每档暴露的失效模式（这才是真正的收获）

| 档 | 失效模式 |
| --- | --- |
| 单 Agent | 最初"做一半就停" → 缺目标持续；加 Goal 后解决，成为当前最佳基线 |
| 共享工作区 | 多 Agent 重读成本；45/75 后 token 熔断 |
| worktree 并行（旧） | ① 混合批次里有写工具→**整批串行**，单批白等 41.5 分钟；② `.npm` 缓存被 `git add -A` 纳入基线→**6 次假冲突**；③ 超时未落盘 |
| worktree 并行（v1 干净） | 修复 ①②③ 后暴露：**一个 Worker 空转 97 分钟、烧掉 53%~56% 的预算**后报错，父 Agent 一直在等它 |
| worktree 并行（v2） | 失控未复现（最大单 Worker 11.3% / 24.8 分钟）；但 tsc 不通过、工具错误从 95 涨到 217、模型仍未用 Reviewer |

## 四、单 Worker 失控的数据

- `worker-call_01_mMhWjunoKfm322Oq`：**error**，指标口径 64,028,171 token / 客户端口径
  67,436,796 token，耗时 **97 分 09 秒**，发出 **1,360 次**模型请求；
- 父 Agent 只有 23 次模型请求，另一个 Worker 在 8 分 01 秒处报错；
- 症状：3 个 worktree 的 commit 号连续 30 分钟不变，`child_events` 最近 400 条里
  `read_file` 占 304 条 —— 典型的"反复调工具、零进展"局部循环。

## 五、为什么"给 Worker 加上限"是错答案

1. 生产环境不能用轮次/时间上限——正常长任务会被一起掐死；
2. 上限治不了根因：空转是**行为问题**（陷进局部循环），不是时长问题，到点死掉不等于走出来。

正确做法是把空转当信号：**检测 → 注入纠偏消息 → 把模型拽出来**。
业界（oh-my-pi 的 `ToolCallLoopGuard`、deepseek-harness 的 `repeat-tool-reminder`）都不设轮次上限。
本项目已实现 Loop Guard v2（见 `todo/loop-guard-v2.md`、`src/core/loop_guard.py`），
真机表现见下一节。

## 六、Loop Guard v2 的真机表现（v2 档）

- **全轮注入 3 次**：父 0 / Scout 1 / Worker 2 —— Scout 第 11 轮 `same_error_family(mild)`、
  Worker 第 39 轮 `same_error_family(mild)`、Worker 第 76 轮 `no_progress(detailed)`；
- **离线回放与真机一致**：`evaluation/replay_loop_guard.py` 在这份归档上重建 22 个子 Agent
  运行（最长 **288 轮**），得到的注入次数、轮次、信号类型与真机逐项对得上
  → 观测改造达到目的，回放判定不漂移；
- **反事实对比**：同一批归档按 v1 口径（`run_command` 算进展）会注入 **≈304 次**，v2 实际只
  注入 **3 次**。那个 288 轮 / 13.6M token 的大 Worker，最长“无新事实”连续只有 **3 轮**——它在
  持续产生新事实（真在做事），v1 会反复报警，v2 的“新事实”语义正确没骚扰它；
- **诚实标注**：上一轮那种“单 Worker 空转 97 分钟”的形态**本轮未复现**，它是偶发的。因此
  **不能把“本轮没出事”讲成“guard 生效了”**——guard 只触发 3 次且都是 mild / 单次，
  **纠偏的救援效果未获证据**。要证明有效，需要一次真正复现失控的运行，或用离线回放做对照实验。

## 七、证据位置

（`evaluation-results/` 被项目忽略，不入库，仅作本地留存）

- 单 Agent：`evaluation-results/big-task-single-goal/`（`result.json` 等）
- 共享工作区：`evaluation-results/big-task-multi-agent-shared/`
- worktree 并行（旧）：`evaluation-results/big-task-worktree-2026-09-25/`（含 `REPORT.md`）
- worktree 并行（v1 干净）：`evaluation-results/big-task-worktree-2026-09-25-rerun/`（含 `REPORT.md`）
- worktree 并行（v2）：`evaluation-results/big-task-worktree-v2-2026-09-25/`（含 `REPORT.md` 与 `loop_guard_replay.json`）
- 无效运行（偶发模型超时）：`evaluation-results/big-task-worktree-2026-09-25-rerun-attempt1-timeout/`
- 无效运行（纠正事件未归档中止）：`evaluation-results/big-task-worktree-v2-attempt1-no-guard-events/`
