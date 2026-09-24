# 指令：带 goal 重跑单 Agent TS 迁移基线

## 背景

无 goal 时，单 Agent 跑到 88 轮就主动停下（只做 26/73 模块）。现在 Goal 机制已就位，本轮让单 Agent 在 goal 驱动下**跑到目标完成或预算耗尽**——这是第一次真正回答"单 Agent 能不能做大任务"。

## 配置

- **任务**：复制 epsilon 到副本（排除 `.venv`/`.git`/`evaluation-results`/`__pycache__`/`.epsilon`），把副本 `src/core` 从 Python 重构成**功能等价**的 TypeScript；harness 从原代码跑，模型只改副本；
- **模型**：`deepseek-flash`（V4.1 Flash），`thinking=high`；
- **委派**：关闭（单 Agent）；
- **goal**：
  - objective：`把副本 src/core 的全部 Python 模块重构成等价的 TypeScript，直到全部模块都有对应 TS 实现且类型检查通过`
  - 预算：`max_rounds=300`、`token_budget=15_000_000`、`time_budget_seconds=3600`
- **不再设人为轮次上限**——轮次由 goal 预算管；
- 评测脚本通过 `GoalPolicy` + `create_goal_tool` + `instruction_message` 接入（照 goal-smoke 的方式），不额外给"做一半就停"的提示，只描述完整任务。

## 记录

1. goal 最终状态（active / complete / budget_limited）、`rounds_started`、`tokens_used`；
2. **进度曲线**（每 25 轮或 2 分钟采样）：排除 `node_modules` 后的 TS 文件数、有对应 TS 的 Python 模块数、该点 `tsc` 状态；
3. `stop_reason`、续跑次数（自动续跑注入了多少轮）、总 token、总耗时；
4. **完成声明是否属实**：评测独立核实最终状态（模块映射 + `tsc`），与模型的 `goal(complete)` 声明对照——对不上就如实记"虚报"。

## 验收标准

1. goal 驱动下，模型**不再"做一段就停"**：要么 `goal complete`，要么 `budget_limited`；
2. 有完整进度曲线（每 25 轮/2 分钟一个点）；
3. 独立核实完成声明的真实性；
4. 原仓库 `src/`、`tests/` 哈希前后一致；
5. 若 73 模块全部有 TS 且 `tsc` 通过 → 记"单 Agent 在预算内完成"；否则记"停在哪个环节、什么原因"。

## 约束

- 跑前报费用确认（token 可能到 10M+ 量级）；
- 副本机制，不碰原仓库；
- 诚实记录，模型虚报完成必须如实标注；
- 只改评测脚本（接入 goal + 进度采样），不改生产代码。

## 结果怎么读

- **单 Agent 被 goal 推到完成**：说明大任务的核心缺口是"目标持续"，多 Agent 不是必需；
- **单 Agent 撞预算仍未完成**：这才是多 Agent 的战场——看多 Agent 能否在同样预算内推进更多文件；
- **单 Agent 又提前声明完成**（虚报）：说明提示词/完成审计还不够强，需要继续加码。

---

转给执行 Agent。这轮结果会第一次告诉我们：单 Agent 在"目标持续 + 无轮次上限"下的真实上限在哪里。
