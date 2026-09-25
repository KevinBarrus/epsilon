# 多 Agent 三档对照：单 Agent / 共享工作区 / worktree 并行（两轮）

同一任务（把副本 `src/core` 的 75 个 Python 模块重构成等价的 TypeScript）、
同 goal、同模型 `deepseek-flash`（`thinking=high`）、同安全熔断口径。
每档只跑一次，是描述性对照，不作统计结论。

## 一、结果（三档，其中 worktree 跑了两轮）

| 档 | token | 墙钟 | 墙钟相对单 Agent | token 相对单 Agent | 工具错误 | 完成度 | tsc | 终止 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 单 Agent（有 goal） | 33,317,726 | 47:24 | 1.00× | 1.00× | 16 | **75/75** | 通过 | Goal complete |
| 多 Agent 共享工作区 | 58,546,994 | 1:56:13 | 2.45× | 1.76× | 69 | 45/75 | 独立复验通过 | 50M token 熔断 |
| worktree 并行（旧，污染） | ≥83,815,202 | ~2:00 | ~2.53× | ≥2.52× | ≥114 | 71/75 | 独立复验通过 | 墙钟到期，收尾异常 |
| worktree 并行（干净重跑） | 120,010,128 | 2:34:33 | **3.26×** | **3.60×** | 95 | 49/75 | 通过（harness） | 120M token 熔断 |

两个比例的算法：分别用该档的墙钟/token 除以单 Agent 的 2844.5 秒 / 33,317,726 token。

## 二、三个可靠结论

1. **多 Agent 的 token 代价稳健更高**：共享档 token 为单 Agent 的 **1.76×**、worktree 档
   **3.60×**（旧轮 ≥2.52× 为下界）。原因是文本通信带来的"重读成本"——子 Agent 拿不到
   父 Agent 的上下文，必须自己再读一遍。**这个成本改不掉**，加并发、加隔离都不影响它。

2. **worktree 隔离 + 并行机制本身可用**：干净重跑里 merge 冲突 **0**（旧轮 6 次全是
   `.npm` 假冲突）、实际并发峰值 **4**、7 个 Worker 干净合并。修掉"混合批次整批串行化"后，
   Worker 确实真并发了。

3. **"并行能不能把墙钟压回来"仍未获干净答案**：干净重跑 2:34:33、只完成 49/75，
   墙钟是单 Agent 的 **3.26×**，比旧轮的 ~2.53× 还差——但主因不是并行本身，而是暴露出的
   新缺陷：**单个 Worker 无上限空转**。

## 三、每档暴露的失效模式（这才是真正的收获）

| 档 | 失效模式 |
| --- | --- |
| 单 Agent | 最初"做一半就停" → 缺目标持续；加 Goal 后解决，成为当前最佳基线 |
| 共享工作区 | 多 Agent 重读成本；45/75 后 token 熔断 |
| worktree 并行（旧） | ① 混合批次里有写工具→**整批串行**，单批白等 41.5 分钟；② `.npm` 缓存被 `git add -A` 纳入基线→**6 次假冲突**；③ 超时未落盘 |
| worktree 并行（干净） | 修复 ①②③ 后暴露：**一个 Worker 空转 97 分钟、烧掉 53%~56% 的预算**后报错，父 Agent 一直在等它 |

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
本项目已按此实现 Loop Guard（见 `todo/loop-guard.md`、`src/core/loop_guard.py`），
下一轮大任务重跑时 `role_loop_guard_injections` 计数会给出纠偏是否生效的直接证据。

## 六、证据位置

（`evaluation-results/` 被项目忽略，不入库，仅作本地留存）

- 单 Agent：`evaluation-results/big-task-single-goal/`（`result.json` 等）
- 共享工作区：`evaluation-results/big-task-multi-agent-shared/`
- worktree 并行（旧）：`evaluation-results/big-task-worktree-2026-09-25/`（含 `REPORT.md`）
- worktree 并行（干净）：`evaluation-results/big-task-worktree-2026-09-25-rerun/`（含 `REPORT.md`）
- 无效运行（偶发模型超时）：`evaluation-results/big-task-worktree-2026-09-25-rerun-attempt1-timeout/`
