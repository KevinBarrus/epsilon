# 指令：Subagent 上下文边界改造（Fork / Fresh 路由 + 缓存继承 + 结构化回传）

## 零、为什么要做（我们踩的坑）

一条外部观点指出：**Subagent 的 handoff 是有损的**——每经一次 handoff 都可能丢失原始证据与推理路径，
于是子 Agent 能力再强也可能只治表象、不治根因。

**这个批评精确命中我们的实现**（代码事实）：

| 现状 | 代码位置 | 问题 |
|---|---|---|
| 子 Agent 只收到**一段字符串** `context` | `src/core/subagent.py:252` | **有损的文本 handoff** |
| 提示词要求父 Agent "把关键路径写进 context" | `src/core/subagent.py:86` 附近 | 让父**手写有损摘要** |
| 子 Agent 是**空上下文** | `_create_spawn_role_tool` 里的 `_run_subagent` | **Fresh**——必须自己重读项目 |
| 回传是**截断的散文** | `src/core/subagent.py:516 _limit_summary` | **上行也有损** |
| 从不设 prompt cache key | `src/core/openai_client.py:345` 只解析命中数 | 无缓存继承 |

→ 直接后果就是我们实测到的：**Worker 重读 21 遍（token 3.6×）+ 接口不对齐（`TS2353`）**。

**但外部观点只看到了 "Fresh Subagent"。参考项目早已支持 Fork**（代码事实）：

- **codex**：`SpawnAgentForkMode { FullHistory, LastNTurns(usize) }`，另有
  `truncate_rollout_to_last_n_fork_turns`（按 user turn 边界截断）；
- **oh-my-pi**：`session.fork()` 克隆 transcript；`providerPromptCacheKeySource: "fork"` →
  **fork 继承父的 `promptCacheKey`**；
- **pi**：`ctx.fork(leafId, options)` / `runtime.fork(entryId, options)`。

**结论**：正确的拆分边界**不是 Role Boundary，而是 Context Boundary**。
本任务把子 Agent 从"只能 Fresh"改成"**可 Fresh 可 Fork**"，并按上下文需求路由。

## 一、目标设计：按"上下文边界"路由

| 角色 | 默认模式 | 判据 |
|---|---|---|
| **Scout**（搜索式探索） | **Fresh** | 会产生**大量临时上下文**，正该就地丢弃 |
| **Worker**（写） | **Fork**（`last_n`） | 需要**完整继承**父的项目理解与冻结接口 |
| **Reviewer**（独立验证） | **Fresh** | 需要**独立性**，不能继承作者的假设 |
| 连续执行（不拆） | 父自己干 | 强耦合、需要连续推理 |

**关键约束**：Fork **不是免费的**——在无状态 API 下，fork = 每请求重发继承的上下文。
所以必须配两条成本控制（参考项目都是这么做的）：

- **`last_n` 截断**（codex）→ 控制继承上下文的**体积**；
- **prompt cache key 继承**（oh-my-pi）→ 控制继承上下文的**单价**。

## 二、实现要求

### 2.1 Fork 机制（父上下文快照 + 截断）

1. 给 spawn 工具新增参数 `mode`：`fresh`（默认）| `fork` | `fork_last_n`；
   `fork_last_n` 另有 `turns` 参数（默认 4，参考 codex 的 `LastNTurns`）。
2. **父上下文快照**：
   - `ToolHandler = Callable[[ToolCall], Awaitable[ToolResult]]` **签名不变**——用
     `contextvars.ContextVar` 传递：`AgentLoop._execute_tool_batch` 在调用工具前，把当前
     `context` 的**快照**写入 ContextVar，spawn handler 从中读取；
   - 注意 asyncio 在 `create_task` 时会复制 ContextVar，因此子 Agent 不会污染父 Agent；
   - **推荐**用 last-user-turn 边界做截断（避免把"半个回合"继承给子 Agent）。
3. **子 Agent 的初始上下文** = 父快照消息 + 角色 system prompt + 任务消息；
   `fresh` 模式保持现状（空上下文 + 任务 + `context` 字符串，向后兼容）。

### 2.2 Prompt cache 继承 + 命中测量

1. 客户端支持可选 `prompt_cache_key` 透传（provider 支持时；DeepSeek 的前缀缓存是自动的，
   此时**关键是保持继承前缀逐字节不变**，不要重排/改写父消息）；
2. `fork` 子 Agent 继承父的 cache key（对应 oh-my-pi 的
   `providerPromptCacheKeySource: "fork"`）；
3. **按子 Agent 记录缓存命中**（`prompt_cache_hit_tokens`），写进 `result.json`，
   **区分 fork / fresh**——这是本改造的核心证据。

### 2.3 结构化回传（修上行损失）

把 `_limit_summary` 的"截断散文"改成**固定小节的结构化回传**（复用已有的
`_limit_required_sections`），例如：

- `CHANGED`：改动的路径清单（**必须是真实存在的路径**，由父侧脚本校验）；
- `EVIDENCE`：跑过的命令与结果（测试/类型检查的通过与否）；
- `BLOCKED`：无法推进的原因（没有则写"无"）。

父 Agent 拿到的是**可验证的证据**，而不是"我觉得我做完了"。

### 2.4 路由引导与配置

1. spawn 工具的 description 要写清**何时用 fork**：
   "需要继承你已读到的项目结构 / 已冻结的接口 → 用 `fork`；只是独立探索或独立验证 → 用 `fresh`"；
2. 角色默认模式可配：`subagent.default_mode`（scout=fresh / worker=fork / reviewer=fresh）；
3. **不要只靠模型自觉**：实测中模型自发用了 21 个 worker、0 个 reviewer——
   harness 必须能**按角色强制**默认模式（配置优先于模型选择）。

## 三、测试（先测后写）

1. `fork` 子 Agent 的初始消息 = 父快照 + 角色提示 + 任务；`fresh` 保持旧行为；
2. `fork_last_n(n)` 按 last-user-turn 边界截断，且 n 超出时退化为全量；
3. ContextVar 隔离：子 Agent 运行不会改变父 Agent 的上下文快照；
4. cache key 继承：fork 请求带上父的 key（provider 支持时）；不支持时前缀**逐字节不变**；
5. 结构化回传：缺 `CHANGED` / `EVIDENCE` 时被补齐或标记；
6. **关键集成测试**（对应验收）：
   - 父先 `read_file("X")`，再 spawn 子 Agent 问"X 里是什么"；
   - **fresh 子 Agent**：必须自己调 `read_file`（有读调用）；
   - **fork 子 Agent**：**不调用 `read_file`** 也能回答，且缓存命中率明显更高。

## 四、验收

1. 上条集成测试通过，且能**量化** fork vs fresh 的 token 与缓存命中差异；
2. 全量测试通过，`fresh` 旧行为无回归；
3. 产出 `evaluation-results/` 下的一份小对照（fork vs fresh 的 token / 命中率 / 读调用次数）；
4. 如实标注：单次运行、描述性结论。

## 五、边界

- **不改 `ToolHandler` 签名**（用 ContextVar）；
- **不设轮次上限**（沿用"长任务不掐死"）；
- Fork 不是"免费无损"——**必须同时落地 `last_n` 截断与缓存继承**，否则只是把重读成本换成了重发成本；
- 不做 KV-cache 层共享（API 够不着），只做**前缀缓存命中**。

---

## 附：与后续实验的关系

本改造完成后，`todo/oncall-read-compare.md` 的多 Agent 档可以升级为
**"fork Scout（继承父已读上下文 + 缓存命中）" vs "fresh Scout"** 的对照——
直接量出 fork 到底省了多少 token。**建议先做本改造，再跑 oncall 实验。**
