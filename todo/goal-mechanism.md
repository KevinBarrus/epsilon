# 指令：实现 Goal 机制（目标持续 / 续跑 / 显式完成 / 预算兜底）

## 背景

单 Agent 大任务实测：模型做一段（26/73 模块）就停下，`stop_reason=completed`，但任务远未完成。根因是 harness 没有"目标"概念——模型不调工具循环就结束，缺目标完成检查与续跑。

参考四家做法（codex `ext/goal`、oh-my-pi `goals/`、deepseek-harness `packages/goal/`、pi 的 steering 队列），共同机制：

1. **goal = objective + 预算**（token / 时间 / 轮数）+ 状态；
2. **续跑由 harness 驱动**：模型停下时若 goal 仍 active，注入一条隐藏续跑消息并开下一轮；
3. **完成必须显式**：模型调用 `goal({op:"complete"})` 才算完成，光停下不算；
4. **预算兜底**：任一预算耗尽进终态，注入"预算用尽、收尾"消息后停。

## 组件（五块）

### 1. `src/core/goal.py`：Goal 模型 + GoalPolicy

```python
@dataclass
class Goal:
    objective: str
    max_rounds: int | None = None
    token_budget: int | None = None
    time_budget_seconds: int | None = None
    status: Literal["active", "complete", "budget_limited"] = "active"
    tokens_used: int = 0
    rounds_started: int = 0
```

`GoalPolicy` 实现现有 `TurnEndPolicy` 协议：

- `follow_up_message()`：
  - `status != "active"` → `None`（结束）；
  - 完成检查通过（见第 5 块的可选 verifier）或模型已标记 complete → `status = "complete"`，返回一个"完成"收尾消息或 `None`；
  - 预算耗尽（rounds / tokens / time 任一）→ `status = "budget_limited"`，返回"预算用尽、收尾"消息；
  - 否则 → `rounds_started += 1`，返回**续跑消息**。
- `observe_usage(usage)`：`tokens_used += usage.total_tokens`（需扩展协议，见第 5 块）。
- `observe_tool_results(...)`：检测模型是否调用了 `goal(complete)`，是则置 `status="complete"`。
- `summary()`：汇报 goal 状态与用量。

**续跑消息内容**（照抄四家的防呆措辞，中文）：

> 继续推进当前目标。目标在跨轮次持续存在，结束本轮**不代表**要把目标缩成现在能做完的较小版本。
>
> 目标：<objective>
> 预算：已用 token <used> / <budget>，已用轮次 <rounds> / <max>。
>
> - **绝不允许把成功重新定义成一个更小、更容易、或已经完成的子集。**
> - 预算耗尽 ≠ 完成，不要因为快没预算就声明完成。
> - 声明完成前必须审计当前仓库实际状态，用直接证据证明目标达成；覆盖范围不足、只间接证据、未检查的"看起来对"，一律视为未完成，继续工作。
> - 只做推进目标的事，不要叙述"我接下来要继续"。

### 2. `goal` 工具

给模型一个显式完成入口（放在 `src/core/goal.py` 或 tools 层）：

```
goal(op="complete")   # 声明目标完成
goal(op="status")     # 查询目标与预算（可选）
```

- 工具 handler 置共享 Goal 的 `status="complete"`；
- 工具只读/内部状态，不需审批。

### 3. `/goal` 命令（`src/core/commands/goal.py`）

- `/goal <objective>`：设定目标（默认预算 max_rounds=50、token_budget=5_000_000）；
- `/goal status`：查看目标、状态、已用预算；
- `/goal clear`：清除目标。

### 4. 持久化

- `session_store` 增加 `type: "goal"` 记录（objective / 预算 / status / tokens_used / rounds_started）；
- `load_messages` 继续跳过非 message；新增 `load_goal` / `append_goal`；
- `Session.restore` 恢复 goal，重建 GoalPolicy。

### 5. AgentLoop 扩展（把用量交给 policy）

- `TurnEndPolicy` 协议增加 `observe_usage(usage: UsageEvent) -> None`；
- `AgentLoop.run` 收到 `UsageEvent` 时调用 `self._end_policy.observe_usage(event)`；
- `WriteVerificationPolicy` 补一个空实现（不破坏现有行为）。

## 完成判定

- **生产**：模型显式 `goal(complete)` + 续跑提示词强制的完成前审计；
- **评测**：GoalPolicy 支持可选 `completion_check: Callable[[], bool]` 硬验证器——通过才算完成（给 TS 迁移这种需要硬判据的任务用）。

## 预算兜底

- 主预算 `max_rounds`（简单、稳健，对标 dsh 的 `maxGoalRounds`）；
- 次预算 `token_budget`、可选 `time_budget_seconds`；
- 任一耗尽 → `budget_limited`，注入收尾消息后结束，不无限循环。

## 测试（先测后写）

1. GoalPolicy：active 时续跑；`complete` 后停；rounds 耗尽停；token 耗尽停；
2. `goal` 工具：调用后 Goal 置 complete；
3. `/goal` 命令：设置 / 查看 / 清除；
4. 持久化：goal 记录往返、`Session.restore` 恢复；
5. AgentLoop：UsageEvent 被传给 policy；
6. 现有 `WriteVerificationPolicy` 测试保持通过。

## 验收标准

- 全量测试通过；
- 冒烟（真机，跑前报费用）：设一个明确目标，**故意构造"模型做一半就停"**的场景，验证 harness 会**自动续跑**直到 goal complete 或预算耗尽，而不是做一段就结束；
- 续跑轮数与预算在会话里可见、可持久化、resume 后恢复；
- 不改变原有单轮（无 goal）行为。

## 约束

- 参考四家、但不照搬：保持 Epsilon 简洁，不引入 SQLite、事件溯源等重设施；
- 先测后写，每完成一块跑一次全量测试；
- 本指令只实现机制，**不跑 TS 迁移大任务**；机制验收通过后，再下"带 goal 重跑单 Agent 基线"的指令。

---

转给执行 Agent。这是把"半途而废"从根上修掉的 harness 能力，做完后我们才第一次能让单 Agent 真正"跑到完成或预算耗尽"，也才能真正测出多 Agent 的价值。
