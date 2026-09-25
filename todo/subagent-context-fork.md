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

## 六、成本模型与临界点（为什么"fork 不免费"）

设：
- `P` = fork 时继承的父前缀（token）
- `D` = fresh 子 Agent 的**发现成本**（找出要读哪些文件 + 读进来 + 试错）
- `G` = 子 Agent 自己的产出性工作（两种模式都要付）
- `k` = 子 Agent 的请求数
- `c` = 缓存命中价 ÷ 未命中价（量级上约 0.1）

| | 一次性成本 | 每请求成本 |
|---|---|---|
| **Fresh** | `D` | 自己的上下文（**也会**被 provider 缓存） |
| **Fork** | 0 | 继承前缀 `P`（命中时 ×`c`） |

**Fork 划算的粗略条件**：`D > k · P · (1 − c)`

→ **`D` 大、`P` 小或命中缓存、`k` 小**时 fork 赢；**`D` 小、`k` 大、`P` 大**时 fork 输。

**代入我们的真实数据**：Worker 平均 **~46k token/请求**、跑 **100~300 请求**。
若无缓存，fork 的重发成本 ≈ `200 × 46k = 9.2M`——**与它自己重读一遍同量级**。
所以：**没有缓存，fork 对我们这种"长子任务"几乎不赚。**

**一个容易忽略的点**：fresh 子 Agent **也会**缓存自己的前缀（它的第 2 个请求起就命中）。
所以 fork 相对 fresh 的缓存优势，主要来自**"fork 的第 1 个请求就已经是命中"**（省掉冷 prefill）——
实测命中率只高约 **0.11**，与此一致。

## 七、微基准实测（结果与它的边界）

`evaluation/fork_fresh_smoke.py`，同一确定性场景（父先读 `docs/architecture.md`，再用三种模式起
Scout 问同一问题），跑两轮：

| 轮次 | 模式 | token | 缓存命中率 | 读调用 |
|---|---|---:|---:|---:|
| 归档（第 2 轮） | fresh | 2,665 | 0.722 | 1 |
| | fork | 2,171 | 0.808 | 1 |
| | fork_last_n | 2,096 | 0.811 | 1 |
| 第 1 轮 | fresh | 1,851 | 0.693 | 1 |
| | fork | 4,682 | 0.836 | 2 |
| | fork_last_n | 1,125 | 0.827 | 0 |

**两条站得住的**：

1. ✅ **缓存继承生效**：fork / fork_last_n 命中率（0.81~0.84）稳定高于 fresh（0.69~0.72），
   两轮方向一致——继承的父前缀确实逐字节命中了前缀缓存。**这是本改造的核心证据。**
2. ❌ **"省掉重读"没有自动发生**：两轮里 fork 子 Agent 仍自己读了一遍目标文件（一次读了 2 遍）。
   根因：**Scout 角色提示词要求"只记录实际读取过的文件"，与 fork 语义冲突**——
   它不信任继承来的读取结果。

**但这个微基准有它测不到的东西（必须说清，否则会误读）**：

- 它问的是"`docs/architecture.md` 里说了什么"——**答案本来就躺在父的上下文里**；
  换句话说：这个任务**无论 fork 还是 fresh 都没有 spawn 的必要**，父自己就能答；
- **Fork 的价值只可能出现在"子 Agent 要做依赖继承上下文的【新工作】"时**；
  若子 Agent 的产出完全可从继承上下文推出，就不该 spawn；
- 所以本微基准是**机制验证**（缓存继承 ✅），**不是价值验证**。

## 八、由此产生的三条设计修正

1. **fork 角色的提示词必须显式放宽引用口径**（安全版本）：
   > 继承上下文里**已经读过的文件可以直接引用，不必重读**；
   > 但**不得引用继承上下文里从未出现过的内容**。

   既让 fork 拿到"省读"收益，又保留防幻觉边界（继承的 tool 结果本身就是 ground truth）。

2. **fork 只应部署在"需要继承上下文 + 要做新工作"的场景**：
   - ✅ 延续式探索（"基于你已读到的架构，找出所有违反它的地方"）；
   - ✅ **Worker 写任务**（继承父的项目图 + 冻结接口，再写代码）；
   - ❌ 不相关分区的扇出扫描（那是 **Fresh** 的地盘）。

3. **oncall 实验的设计要改**（原来提的"fork Scout vs fresh Scout"是错的）：
   - oncall 是**不相关分区的扇出扫描** → **Fresh 才是正解**；该实验应测**原假设**：
     **单 Agent（串行读） vs fresh Scout 扇出（并行读）**；
   - 若仍想测 fork，必须**加一档"父先做项目测绘 → fork Scout 深读各分区"**（延续式探索），
     而不是让 fork 去干 fresh 的活。

---

## 附：与后续实验的关系

本改造完成后：

- `todo/oncall-read-compare.md` 的主档应测 **单 Agent vs fresh Scout 扇出**（原假设）；
- 想测 fork 的价值，需要**另一个场景**：要么"延续式探索"，要么**重构任务的 v3**
  （父先冻结接口 → fork Worker 实现）——那才是 fork 的主场。
