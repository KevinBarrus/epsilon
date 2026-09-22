# 问题 30：多 Agent（SubAgent）设计

本文档为多 Agent 能力的设计稿，只做设计，不含实现。经审阅后再进入编码。

## 一、背景与目标

### 背景

Epsilon 目前是单 Agent：一次会话里一个模型负责"读代码 → 改文件 → 跑测试"的全过程。问题在于，探索、检索这类工作会产生大量中间内容（搜索结果、日志、文件片段），它们会涌入主上下文，占用 token 且后续不再引用。

多 Agent 的解法：把这类工作**委派给子 Agent**，子 Agent 在自己的上下文里独立完成，只返回有界摘要，主上下文保持干净。这正是 oh-my-pi（task 工具）、codex（spawn_agent 工具）、ClaudeCode（subagents）三家共同的做法。

### 目标

- **第一版**：只实现只读 Scout，验证上下文隔离是否能减少主 Agent 的探索噪声，并分别观察父 Agent token、全部 Agent token、耗时和任务结果；
- **第二版**：并行加速、异步交付、自定义 subagent 等进阶能力。

### 定位

方案五（P1）已把多 Agent 从"可评测"降级为"**可演示**"：D 档只跑一遍，结果只作描述性观察，不宣称具有统计意义。父 Agent token 与全部 Agent token 分开报告，不能用父上下文变小推导总成本下降。

## 二、核心机制（第一版与第二版共享）

### 2.1 spawn_agent 是一个工具

主模型"知道"并能调用子 Agent 的方式，与它调用 read_file、run_command 完全相同——把 `spawn_agent` 注册成工具。三个配合点：

1. **工具注册**：`spawn_agent` 进入 ToolManager，description 写明"把只读探索任务委派给 Scout，Scout 独立工作、只返回有界摘要"；
2. **系统提示词注入说明**：说明 Scout 的职责、只读边界和适用场景，供主模型判断何时委派；
3. **参数 schema**：第一版只有 `task`（任务描述），不为尚不存在的角色保留字段。

主模型调不调、何时调，由它自己判断——这正是工具机制的天然属性，与"on 只是允许、不是强制"的语义一致。

### 2.2 on/off 的实现

- **off（默认）**：`model_tools()` 过滤掉 `spawn_agent`，系统提示词不注入 Scout 说明，主模型看不到、不会调用；
- **on**：工具对模型可见，并注入 Scout 说明，主模型可能调用；
- 开关只在当前进程内生效，恢复会话后重新回到 off。第一版不为一个布尔值增加持久化状态。

### 2.3 通信方式：工具调用 + 结果回传

父子之间**没有**消息队列、没有共享内存、没有来回聊天。完整闭环：

```
主模型调用 spawn_agent(task="定位 XX 逻辑")
   → Agent Loop 执行工具：启动临时 Scout 运行时
   → 子 Agent 独立上下文里跑（读文件、搜代码）
   → 子 Agent 结束，产出 summary
   → summary 作为 spawn_agent 的"工具结果"返回
   → Agent Loop 把 summary 写进父上下文
   → 主模型下一轮看到 summary，继续
```

- **父发一次任务，子回一次结果**，中间无来回；追问只能再 spawn 一次；
- 摘要包含结论、相关文件、关键证据和建议下一步，设置固定字符上限，不要求字面上的一句话；
- 第一版**同步**：主模型调 spawn_agent 后阻塞等子 Agent 跑完。

### 2.4 共享与不共享

| | 共享 | 不共享 |
|---|---|---|
| 对话历史 / 上下文 | ❌ | ✅ 子 Agent 独立上下文，看不到父的完整对话 |
| 工作区（文件系统） | ✅ 父子读取同一个工作区 | |
| 项目指令 | ✅ Scout 启动时重新加载 | |
| Artifact Store | | ✅ 第一版不跨 Agent 共享，Scout 必须把结果写进摘要 |
| 工具集 | | ✅ Scout 只有只读工具，且没有 `spawn_agent` |

### 2.5 reasoning_content 不跨会话

DeepSeek 思考模式 + 工具循环要求回传 reasoning_content，但这条规则**只在会话内**生效：

- 子 Agent 会话内多轮工具循环时，reasoning 回传照常（复用方案一的 AgentLoop 自动继承）；
- **跨 Agent（父子 / 子子）不传 reasoning，只传 summary**。summary 在父会话里是 `role: tool` 的工具结果，不是 assistant 消息，不触发回传规则；
- 因此不存在"多个 subagent 互相传思考链导致 token 爆炸"的问题。

## 三、第一版方案

### 3.1 只实现 Scout

第一版只有一个内置角色：Scout（侦察）。它负责找文件、搜索代码、阅读实现和定位问题，只拥有 `read_file`、`list_files`、`search_files` 三个只读工具。

第一版不实现 Worker 和 Reviewer。写操作审批、共享工作区冲突、测试命令限制和文件隔离统一留到第二版讨论。

### 3.2 `/subagent` 命令：on/off 开关 + 展示

- 进入后展示 Scout、当前开关状态、继承的模型与思考强度；
- 提供 `on` / `off` 切换；
- **不做**运行时增删改 subagent；
- Scout 统一显示"继承主模型"，余额沿用主模型的现有展示，不单独查询。

### 3.3 并发策略

`spawn_agent` 固定为 `execution_mode = parallel`，直接复用现有只读工具批次并行机制。主模型可以在一次响应里发起多个互不依赖的 Scout，结果仍按原工具调用顺序写回父上下文。

第一版没有写子 Agent，因此不需要动态执行模式、写锁或 worktree。

### 3.4 模型与额度

- Scout **继承主模型和思考强度**，不单独配置模型；
- 不做余额降级。Scout 与主 Agent 使用同一模型配置，不存在可降级的备用配置。

### 3.5 并发与深度限制

- **深度限制 = 1**：Scout 的工具集中没有 `spawn_agent`，因此不能创建孙 Agent；
- **并发上限 = 3**：使用第一版固定常量，不增加配置项；
- 每个 Scout 使用固定工具轮次上限和总超时；父请求取消时同步取消仍在运行的 Scout；
- Scout 失败、超时或达到工具轮次上限时，`spawn_agent` 返回结构化错误，父 Agent 可以继续处理任务。

### 3.6 通信落盘

`spawn_agent` 的 task 作为 assistant 消息的 tool call 落盘，Scout 的摘要作为 tool 结果落盘。恢复会话后可以看到父 Agent 委派了什么、Scout 返回了什么。

第一版不把 Scout 的完整内部轨迹、轮次、token 和耗时写进父会话 JSONL；这些运行指标只进入评测事件，避免为可演示功能扩展 Session 持久化协议。

### 3.7 评测开关

用 `/subagent` 开关做一题长任务的"开 / 关"冒烟对照，分别记录：

- 任务是否完成；
- 父 Agent 实际 token；
- 所有 Agent 实际 token 总和；
- 总耗时；
- Scout 调用次数；
- 父上下文新增工具结果的字符数。

只报告这次运行的描述性结果，不宣称总 token 一定下降，也不据此给出稳定性能结论。

## 四、第二版方案

### 4.1 worktree 文件隔离（并行写）

目标场景：多个 Worker 并行写文件不冲突。采用 oh-my-pi 模式：

- 每个写子 Agent 在 copy-on-write worktree 里跑，跑完 `mergeIsolatedChanges` 合回主工作区；
- **前提**：工作区必须是 git 仓库。评测工作区是无 git 快照，需要先解决"给快照 git init"的矛盾；
- **冲突策略**：merge 失败保留分支、报 conflict（oh-my-pi 的 rescueTaskBranch 模式），不让冲突静默丢改动。

### 4.2 异步交付

主 Agent 不必阻塞等子 Agent。采用 oh-my-pi 的 `agent://<id>` 交付模式：

- 子 Agent 后台跑，结果落到 `agent://<jobId>`；
- 主 Agent 先回复用户"正在等待子 Agent"，子 Agent 返回后结果在主 Agent 下一次工具调用前注入。

### 4.3 分角色配模型（省钱场景）

配置文件的 `model` 字段启用：Scout 用便宜模型、Worker / 主 Agent 用贵模型，或跨服务商分摊额度。**第一版不做，留字段占位。**

### 4.4 自定义 subagent

允许用户定义自己的 subagent（name / description / system prompt / 工具权限 / 模型）。形式（UI 还是配置文件）届时再权衡"对懒用户是否友好"，不在第一版决定。

## 五、验收标准

### 第一版

- `/subagent on` 后主模型能通过 `spawn_agent` 工具委派任务，`off` 后看不到该工具；
- Scout 只拥有只读工具，在独立上下文工作并返回有界摘要；
- 一轮最多并行三个 Scout，返回顺序与工具调用顺序一致；
- 父请求取消、Scout 超时和工具轮次上限均能正确收束；
- task 与摘要随父会话落盘，恢复后完整；
- Scout 无法再调用 `spawn_agent`；
- 一题长任务冒烟通过，开/关两档按约定口径输出描述性指标；
- 全量单元测试通过。

### 第二版

- worktree 隔离下多个 Worker 并行改文件、合并结果正确、冲突有明确处理；
- 异步交付下主 Agent 不阻塞，结果能正确注入；
- 分角色配模型生效。

## 六、与三家的对照（取舍依据）

| 项目 | 子 Agent 触发 | 上下文隔离 | 文件隔离 |
|---|---|---|---|
| oh-my-pi | task 工具显式调用 | 独立 session + 结构化输出 | 默认 worktree（依赖 git，可关） |
| codex | spawn_agent 工具显式调用 | 独立 Thread | 无（共享 cwd + 并发槽位） |
| ClaudeCode | 自动委派 + 显式 | 独立 context window，只回 summary | 可选（--worktree flag） |
| **Epsilon（本方案）** | spawn_agent 工具显式调用 | 独立 context，只回有界摘要 | 第一版只读共享，第二版待对齐 |
