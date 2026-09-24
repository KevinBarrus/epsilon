# 指令：修正 token 计账 + 无人工预算重跑单 Agent

## 背景与共识

上一轮 goal 记到 14,597,580 token，实际总量 15,013,458——**漏算了上下文压缩请求的 token**。同时我们确认了一个原则：

- **不做"同预算谁先完成"的比较**（无意义：真正要比的是**总 token、总耗时、失败次数**）；
- **预算不设产品/评测上限**，只保留一个**极高的安全熔断**（防死循环烧钱），任务让它**跑到自然结束**（`goal complete`）或熔断；
- 因此 **token 计账必须准确**——它是主指标，账错则比较全废。

## 任务 1：修正 token 计账（必须）

**根因**：`GoalPolicy.observe_usage` 只在 AgentLoop 收到 `UsageEvent` 时被调用；而**压缩请求**（`ContextManager.generate_context_summary` → `client.stream_chat`）不走 AgentLoop 事件流，**子 Agent 请求**也走各自独立的 AgentLoop——两者的 token 都没进 goal 的账。

**修法（在客户端边界统一计账）**：

- 让所有模型请求（主循环 + 压缩 + 子 Agent）的用量，都汇入同一个"用量源"；
- `GoalPolicy` 从这个用量源读取总量（或由运行时在每次请求后喂给它），**不再只依赖 AgentLoop 的事件**；
- 无论走哪条路径，`goal.tokens_used` 必须等于**全部请求 total_tokens 之和**。

**测试**：构造一次会触发压缩的运行，断言 `goal.tokens_used == 所有请求的 total_tokens`（含压缩请求）。

## 任务 2：预算改为"默认无上限 + 极高安全熔断"

- `/goal` 的 `token_budget` **默认 None（无上限）**；`max_rounds` 默认 None；`time_budget_seconds` 默认 None；
- 保留三者作为**可选**参数（用户/评测想设才设）；
- 评测脚本用**极高的安全熔断**（建议 `token_budget=50_000_000`、`time_budget_seconds=7200`），**只为防跑飞，不作为比较参数**；
- 真实用户场景：默认无上限 + 用户可随时中断（`Esc` 取消）。

## 任务 3：重跑单 Agent（无人工预算）

- 任务：复制 epsilon 到副本，把副本 `src/core` 从 Python 重构成等价 TypeScript；harness 从原代码跑；
- 模型：`deepseek-flash`，`thinking=high`；委派关闭（单 Agent）；
- goal objective：`把副本 src/core 的全部 Python 模块重构成等价的 TypeScript，直到全部模块都有对应 TS 实现且类型检查通过`；
- 预算：**只有安全熔断**（50M / 2 小时），**跑到 `goal complete` 或熔断**；
- 进度采样：每 25 轮或 2 分钟一次——排除 `node_modules` 的 TS 文件数、模块映射数、`tsc` 状态、**累计失败次数**（工具错误 + 重试 + tsc 失败）。

## 记录（本轮主指标）

1. **准确的**总 token（必须等于客户端全部请求之和）、总耗时；
2. **失败次数**：工具错误数、重试数、各采样点 tsc 失败；
3. goal 最终状态（complete / 熔断）、续跑次数、模型请求数、工具批次数；
4. 进度曲线；
5. 完成声明的真实性核实（模型 `goal(complete)` vs 独立检查的模块映射 + tsc）；
6. 原仓库 `src/`、`tests/` 哈希前后一致。

## 验收标准

1. goal 的 `tokens_used` == 客户端全部请求 token 之和（**计账准确**，含压缩请求）；
2. 运行**跑到自然结束或熔断**，不因人为小预算中止；
3. 有完整进度曲线与失败次数统计；
4. 独立核实完成声明；
5. 原仓库零改动。

## 约束

- 先修计账、补测试，再跑真机；
- 跑前报费用确认（无小预算后可能到 30M+ 量级）；
- 诚实记录，虚报/失败如实标注；
- 不改其他生产逻辑。

---

转给执行 Agent。这轮之后，"单 Agent 跑到自然结束需要多少 token / 多久 / 多少次失败"就是干净基线了，之后的单 vs 多对比直接看这三个数，不再有"人为上限"的污染。
