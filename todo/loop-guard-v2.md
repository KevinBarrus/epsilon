# 指令：Loop Guard v2 + 可回放观测 + 真机验证（一次做完）

> 用户已出门，**预先授权本轮付费重跑**。执行方按本文三阶段一次做完：
> **Phase 1 观测 → Phase 2 检测增强 → Phase 3 真机重跑**。
> 任何阶段测试不过，**不许进入下一阶段**。

## 零、为什么要做（v1 的三个失效点，数据支持）

v1 已在真实轨迹上暴露三个失效点：

1. **B 口径过松 → 提醒泛滥被忽略**（最关键）。按 v1 口径（`run_command` 算进展）在真实
   归档上回放：≤8 轮的连续"无进展"串有 **97 段**，B 会注入 **≈304 次**；若简单把
   `run_command` 剔出进展，最长无进展串变 **1510**、B 会注入 **≈744 次**——**比 v1 更糟**。
   → 结论：**光改"进展定义"不够**，必须同时改 B 的**语义**与提醒的**节制**。
2. **A 不可验证**：归档没有参数，无法判断重复调用是否真的同参。
3. **无 Worker 标识**：连"这些串属于哪个 Worker"都无法归因。

## 一、Phase 1：可回放观测（免费，先写测试）

目标：**每条子 Agent 轨迹能被精确离线回放**，重跑 A/B/新增信号的判定。

1. **整轮批次记录**（关键，否则"按轮算 B"永远不准）：
   子 Agent 已在发 `ToolBatchEvent`，落盘为
   `{"type":"batch","agent_run_id":…,"round":n,"execution_mode":…,"calls":[{"call_id","tool","args_digest","args_preview"}]}`
2. **逐条结果补字段**：已有的 result 记录补上 `call_id`、`args_digest`、`agent_run_id`。
3. **Worker 标识**：给 `ToolExecutionEvent` / `ToolCallStartedEvent` / `ToolCallCancelledEvent`
   加 `agent_run_id: str = ""`（带默认值，向后兼容），由 `AgentLoop(run_id=…)` 透传；
   `_create_spawn_role_tool` 把父 Agent 发起 spawn 的 `tool_call.call_id` 作为子 `run_id` 传下去。
4. **签名摘要复用一份实现**：直接调用 `loop_guard` 里的规范化参数函数算 `args_digest`，
   禁止另写一套（两套口径必然漂移）。
5. `args_preview` 截断 **≤120 字符**（人读 + 报告用）。

**决策（执行方三问）**：① 用"给事件加字段"，**不用 wrapper**；② **要** `args_preview`；
③ **要** batch 整轮记录。

## 二、Phase 2：检测增强（v2）

### 2.1 检测 A 增强：补两种信号

- **`abab_action_cycle`**：算每一轮的"动作批次指纹"（该轮所有调用签名排序后取哈希）；
  若最近 4 轮的指纹呈 **A,B,A,B**（A≠B）→ 触发。抓"改测试→跑→改回去→再跑"这类交替循环。
- **`same_error_family`**：错误族 = 结构化错误码；无码时取规范化后的错误首行。
  连续多轮出现**同一错误族**即触发，**换了调用也抓**（v1 只看调用签名，换调用即漏）。

### 2.2 检测 B 重设计（**本次最关键**）

v1 的 B = "连续 N 轮没有成功的非读调用" → 在"合法探索期"（一直在读新文件、还没开始写）
就会狂响。**改为"是否产生新事实"**：

- **新事实（new fact）** = 满足任一：
  - 出现一个**此前未见过的动作签名**（`tool + args_digest` 不在本次 run 的已见集合里）——
    覆盖"读了新文件 / 跑了新命令 / 换了新参数"；
  - 一次**成功的写/编辑/委派**（`file.write` / `agent.*`）；
  - `run_command` 产生了**新的输出指纹**（同命令但输出变了，也算新事实）。
- **B 触发**：连续 `no_progress_rounds`（默认 8）轮**没有任何新事实**。
- 语义变化：**"只要没写文件"不再报警**，只有"**原地打转、读取集合不再增长**"才报警。

**进展定义同步修正**：`run_command` **不再默认算进展**（只有产生新签名/新输出指纹才算）。

### 2.3 提醒节制（**必须落地，否则 744 次洪泛**）

- **每轮最多注入一条提醒**（硬上限）；多条候选时按**优先级**挑一条：
  **`进展不变 > 错误族 > 重复调用 > abab`**；
- 阈值分级沿用 v1（`repeated_call` 用 3/5/8；`no_progress` 用 `no_progress_rounds`）；
- 提醒文案**加"防持久化"那一句**：明确告诉模型"这是本轮临时提醒，不是用户偏好，
  **不要写进 Memory / Skills 或任何持久文件**"。

### 2.4 离线回放工具

- 新增 `evaluation/replay_loop_guard.py`：读新格式的 `child_events.jsonl`（含 batch/args_digest/
  agent_run_id），**按 run_id 分组**逐轮喂给 guard，输出：
  每个 run 的**首个触发轮次、各信号触发次数、注入次数**。
- **验收**：用 Phase 3 重跑产出的归档能跑通，并给出"guard 在该轨迹上触发了几次、第几轮触发"。

## 三、Phase 3：真机重跑（已预授权，付费）

- **安全上限沿用现有熔断：120M token / 3 小时墙钟**（不因本次改动放宽）；
- 副本机制不变，**原仓库零改动**；
- 命令（先 `--prepare`，再执行）：
  ```
  uv run python -m evaluation.big_task_single_goal --confirm --delegate \
    --isolate-workers --source-core /tmp/epsilon-single-goal-iq3v424t/workspace/src/core
  ```
- **观测重点**（写进 REPORT.md）：
  1. `role_loop_guard_injections`：父 / Worker / Reviewer 各触发几次；
  2. 触发**轮次**与**信号类型**（A / B / abab / 错误族）；
  3. 是否还出现"**单个子 Agent 长时间无新事实**"（用 Phase 2.4 回放验证）；
  4. 墙钟与完成度（回答"并行能否压缩墙钟"）；
  5. 结果与轨迹归档到 `evaluation-results/`（`evaluation-results/` 被忽略，不入库）。
- **如实记录**：若本轮**没有复现**那个失控（它是偶发的），就写"未复现，v1/v2 的实际纠偏效果
  本轮未获证据"，**不许把"没出事"讲成"guard 生效了"**。

## 四、测试与验收

1. Phase 1：batch/字段/`agent_run_id` 落盘正确；向后兼容（旧事件无 `agent_run_id` 不报错）；
2. Phase 2：A 增强（abab、错误族）、B 重设计（新事实语义）、**每轮最多一条 + 优先级**、
   防持久化文案——逐条单测；
3. **回归**：全量测试通过；v1 的 goal / 混合批次机制无回归；
4. Phase 2.4 回放工具可在归档上跑通；
5. Phase 3 产物齐全（REPORT.md + result.json + 归档）。

## 五、边界（明确不做）

- **不设轮次/时间上限**（沿用"长任务不掐死"原则）；
- 不自动改写模型调用（advisory，只加消息）；
- 不为 Phase 3 放宽熔断；
- 全链路 fail-open：检测/落盘失败绝不影响工具执行与本轮运行。

---

## 附：与 v1 的关系

`todo/loop-guard.md` 第九节是**方向**；本文是**可执行规格**，冲突时以本文为准。
v1 的代码不回滚，v2 在其上增量改造（新增信号、重设计 B、加节制）。
