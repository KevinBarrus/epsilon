# 指令：重跑单 Agent 大任务（去掉轮次 cap + 吞吐量指标）

## 背景

上一轮单 Agent 跑 Python→TS 重构，停在 `max_tool_rounds: 150` 这个**人为 cap** 上（`stop_reason: tool_limit`），而不是自然失败。数据因此作废——无法判断单 Agent 本来会不会完成、会不会撞上下文。

更关键的发现：`context_overflow: False`，期间 3 次压缩。**单 Agent 的上下文管理是工作的，它没有爆上下文**。所以"单 Agent 因上下文爆掉而失败"这个假设不成立；它的真实问题是"串行太慢"（150 轮只做了 23/73 个文件）。

本轮目标：拿掉人为 cap，让单 Agent 跑到自然结束，并用**吞吐量曲线**记录它的进度，作为多 Agent 的对照基线。

## 改动 1：去掉轮次 cap，改成本/时间安全网

- **不再用 `max_tool_rounds` 作为主停止条件**（不要 150，也不要换成另一个拍脑袋的轮数）；
- 改用**安全网**防止跑飞：总量上限建议 `30M token` 或墙钟 `60 分钟`（可调），触到才停；
- 记录 `stop_reason`，必须区分：
  - `model_completed`（模型自己结束）；
  - `token_budget` / `time_budget`（撞安全网）；
  - `context_overflow`（真的爆上下文）；
  - `model_claimed_finished`（模型**自称完成**——必须单独标记，因为它可能虚报）。

## 改动 2：吞吐量指标（本轮核心）

每 **25 轮或每 2 分钟**采样一次进度，写进 `result.json` 的 `progress` 数组：

- 当前轮次、已耗时；
- **排除 `node_modules` 后**的 TS 文件数（上一轮统计把 node_modules 也算进去了，是 bug，本轮必须先排除）；
- 有对应 TS 的 Python 模块数（复用 `modules_without_matching_ts` 逻辑）；
- 该采样点的 `tsc` 是否通过（采样时跑一次类型检查）。

据此算出三个对照指标：

- `files_per_100_rounds`；
- `files_per_minute`；
- 结束时的 `tsc_pass` 与 `py_modules_converted / 73`。

## 改动 3：重跑单 Agent

- 同样任务（把副本 `src/core` 从 Python 重构成 TypeScript）、同样模型（`deepseek-flash` = V4.1 Flash）、`thinking=high`；
- **委派关闭**（`delegation_tools_registered: False`）；
- **副本机制**：复制 epsilon（排除 `.venv`/`.git`/`evaluation-results`/`__pycache__`/`.epsilon`），模型只改副本，harness 从原代码跑；
- 全程记录 `events.jsonl`。

## 验收标准

1. `stop_reason` 明确，是自然结束还是安全网，**不再有人为轮次 cap**；
2. 有完整进度曲线（每 25 轮/2 分钟一个采样点）；
3. 排除 node_modules 后的准确 TS 文件数与 Python 模块映射数；
4. 若 73 个 Python 模块全部生成对应 TS 且 `tsc` 通过 → 记录"单 Agent 在该预算内可完成"；
5. 若未完成 → 记录"在哪个文件/环节停下、什么原因"；
6. 原仓库 `src/`、`tests/` 源码哈希前后一致；
7. 费用在跑前报用户确认（无 cap 后 token 可能到 20M+ 量级）。

## 约束

- 只改评测脚本（cap → 安全网 + 进度采样），不改 Epsilon 生产代码；
- 诚实记录：模型自称完成但实际未完成，必须如实标注；
- 本轮只跑**单 Agent 对照**，跑完停下等指令，再跑多 Agent 实验。

---

转给执行 Agent。单 Agent 基线拿到后，我们再用同一套吞吐量指标跑多 Agent，看"多 Agent 是否在同样时间/轮次内推进更多文件"——这才是判断多 Agent 价值的正确比较。
