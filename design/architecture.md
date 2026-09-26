# 整体架构设计

## 1. 项目定位

epsilon 是一个结构清晰、可恢复、可扩展的 Python Coding Agent Runtime。

第一阶段的目标是把一次 Coding Agent 会话做好，并让整个运行过程能够被理解、解释、测试和恢复。项目暂时不追求同时实现完整的产品级 Runtime、长程控制面、自进化平台和复杂控制论系统。

epsilon 的核心职责是：

- 接收用户任务；
- 调用模型；
- 管理上下文；
- 执行工具；
- 处理权限和错误；
- 保存会话历史；
- 在上下文和工具边界内持续运行。

Skills、Memory、Evaluation、SubAgent 和长程 Goal/Todo 控制面作为后续扩展逐步加入。

## 2. 总体架构

```text
CLI / TUI
   │
Application Layer
   │
Agent Runtime
   ├── Session
   ├── Agent Loop
   ├── Context Manager
   ├── Tool Manager
   └── Model Client
   │
Persistence Layer
   ├── Session JSONL
   ├── Compaction Record
   └── Pending Message Log
   │
Extension Layer
   ├── Skills
   ├── MCP
   ├── Memory
   ├── Evaluation
   └── SubAgent
```

主要依赖方向：

```text
CLI / TUI
    ↓
Application
    ↓
Runtime
    ↓
Tools / Model / Context / Persistence
    ↓
Filesystem / External API
```

上层可以依赖下层，下层不能反向依赖 CLI、TUI 或具体产品展示逻辑。

## 3. 核心运行时层次

### 3.1 Session

`Session` 表示一个可恢复的 Agent 会话，负责保存：

- `session_id`；
- 当前工作目录；
- 对话历史；
- 上下文压缩记录；
- 消息与压缩记录的持久化降级状态。

模型配置由 `Settings` 持有，权限配置由 `ToolManager` 和 `PermissionManager` 持有。Session 是长期存在的会话与持久化边界，不负责具体模型请求、工具实现或活动请求状态。

### 3.2 Turn

Turn 表示一次用户请求从开始到结束的**概念边界**。v1 中它对应一次 `AgentLoop.run()` 调用，状态保留在当前 `ui.py + AgentLoop` 调用栈中，不作为独立对象或持久化记录。

一个 Turn 可能包含多次模型请求和工具调用，例如：

```text
用户请求
  → 模型请求
  → 工具调用
  → 工具结果
  → 模型继续请求
  → 更多工具调用
  → 最终回答
```

取消、错误和完成状态由 AgentLoop 返回结果与 Session 消息状态表达；Token 用量暂由上下文估算和评测层记录，不承诺跨进程恢复活动 Turn。

### 3.3 Step

Step 表示 Turn 中一次具体的模型请求及其响应。v1 仅将它作为 AgentLoop 内部循环边界，不创建独立类。需要断点续跑、并发子任务或逐 Step 审计时，才会增加可持久化的 Turn/Step 记录。

## 4. Agent Loop

Agent Loop 是最核心的执行模块，负责模型和工具之间的循环：

```text
准备当前上下文
    ↓
请求模型
    ↓
解析模型响应
    ├── 普通文本 → 记录响应
    ├── 工具调用 → 执行工具
    └── 错误/中断 → 处理或结束
    ↓
工具结果写回上下文
    ↓
判断是否继续请求模型
```

Agent Loop 只负责决定下一步执行流程，不负责：

- 实现具体工具；
- 持久化所有历史细节；
- 决定 CLI/TUI 如何展示；
- 实现记忆检索；
- 管理复杂的多 Agent 图；
- 承担长期项目控制面职责。

第一版采用简单、显式的模型—工具循环，优先保证可读性和可测试性。复杂的控制器、预测策略和长程调度不直接放入第一版 Agent Loop。

## 5. Model Client

Model Client 负责与模型服务通信，第一版优先支持 OpenAI-compatible API。

主要职责：

- 构造模型请求；
- 支持流式响应；
- 解析文本、工具调用和结束原因；
- 处理基础重试和 API 错误；
- 记录 Token 使用量和请求元数据。

建议通过稳定的协议隔离模型实现：

```python
class ModelClient(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> AsyncIterator[ModelEvent]:
        ...
```

Model Client 不负责执行工具，也不直接修改 Session。Agent Runtime 接收它产生的模型事件后，再决定如何更新状态。

## 6. Context Manager

Context Manager 负责把内部会话状态转换为一次模型可见的上下文。它不只是保存 `messages`，还负责：

- 生成模型请求消息；
- 注入系统指令和工作区信息；
- 计算近似 Token 使用量；
- 管理上下文预算；
- 截断过大的工具结果；
- 判断上下文是否接近上限；
- 执行历史压缩；
- 记录压缩边界和摘要。

第一版优先实现三项能力：

1. Token 近似计算；
2. 大工具结果截断；
3. 超过阈值后压缩较旧历史。

内部历史和发送给模型的 Prompt 视图应当分开。持久化历史可以比当前模型输入更完整，模型输入则需要根据当前模型能力、Token 预算和工具结果限制进行转换。

上下文压缩可以借鉴 Codex 和 MiniCode-Python，但第一版不引入 PID、Kalman 或多控制器上下文调节。先保证静态阈值压缩可靠，再根据实际运行数据决定是否需要动态策略。

## 7. Tool Manager

工具系统分为三个部分：

```text
ToolDefinition：名称、描述、参数 Schema、能力元数据
ToolExecutor：具体执行逻辑
ToolManager：注册、查找、调用、结果处理
```

第一版工具范围：

- `read_file`；
- `list_files`；
- `search_files`；
- `write_file`；
- `edit_file`；
- `run_command`。

每个工具都应具有：

- 输入校验；
- 执行超时；
- 结构化错误结果；
- 输出截断；
- 是否需要确认的属性；
- 只读、有副作用或高风险等能力元数据。

工具执行流程：

```text
Tool Call
  → 查找工具
  → 参数校验
  → 权限判断
  → before hook
  → 执行与超时保护
  → after hook
  → 结果截断
  → 写回 Tool Result
```

工具定义与运行时分离的设计参考了 Codex；Hook、超时和结果限制参考了 MiniCode-Python。工具系统暂时不追求复杂的动态发现和并发编排，先确保基础文件与命令工具可靠。

## 8. Permission Manager

文件读写和命令执行不能直接绕过权限。第一版至少提供三种权限模式：

- `read-only`：只允许读取和搜索；
- `workspace-write`：允许修改当前工作区；
- `full-access`：允许执行更高风险命令，需要明确确认。

权限判断由独立模块负责。具体工具只声明自己需要的能力，不在每个工具内部各自实现一套权限规则。

后续可以加入：

- 单次工具确认；
- 命令前缀授权；
- 工作区路径限制；
- 网络访问控制；
- 更细粒度的沙箱。

这些能力应在基础工具可用后逐步增加，而不是第一版同时实现所有平台安全机制。

## 9. Persistence

第一版采用 JSONL 保存 Session 历史：

```text
.epsilon/
└── sessions/
    ├── <session-id>.jsonl
    └── .<session-id>.pending.jsonl
```

主 JSONL 每行保存一条 `Message` 或 `CompactionRecord`：

```json
{
  "role": "assistant",
  "content": "...",
  "tool_calls": []
}
```

压缩记录使用 `type: "compaction"`，工具调用嵌入 assistant 消息、工具结果使用 `role: "tool"`。主日志最终写入失败时，消息暂存至同目录 pending JSONL 并在下次恢复时迁移。

v1 不持久化以下运行期事件：

- `turn_started`、`turn_finished`；
- 活动请求状态；
- 单次模型请求元数据和 Token usage；
- 独立的工具调用事件。

保存完整消息与压缩记录而不是只保存最终 Prompt，有利于：

- Session 恢复；
- 调试模型和工具行为；
- 重建上下文；
- 生成运行轨迹；
- 后续评测和记忆提取。

第一版不引入 SQLite、复杂索引和 Rollout 压缩。等历史查询、并发写入或数据规模成为真实问题后，再考虑引入 SQLite 或其他索引层。

## 10. Extension Layer

Skills、MCP、Memory、Evaluation 和 SubAgent 都属于扩展能力，但它们不能直接破坏 Runtime 的边界。

### Skills

负责加载项目级和用户级 Skill，解析元数据，并将被选择的 Skill 转换为模型上下文或工具依赖。Skill 加载、选择和注入应与 Agent Loop 分离。

### MCP

负责连接外部 MCP Server，并将外部工具或资源适配成 epsilon 内部协议。MCP 的连接和错误不应污染核心 Agent Loop。

### Memory

当前 `memory.py` 仅保存当前进程内单个 Session 的消息历史，是 Session 运行期缓存，不是长期记忆。它不负责跨会话事实、用户偏好、检索、注入或写回。

长期记忆属于后续扩展；是否采用文件型事实、关键词检索或其它机制，必须在真实跨会话需求出现后单独设计。

### Evaluation

负责记录完整运行轨迹、验证任务结果、统计成功率、Token 和工具错误。可以借鉴 Bayesian-Agent 的 Trajectory Evidence，但评测层不应反向控制核心 Runtime 的内部实现。

### SubAgent

后续通过独立的子 Session 实现。子 Agent 应有独立历史、任务边界和完成结果，父 Agent 通过结构化消息接收结果，而不是共享一份可变消息列表。

## 11. 第一版范围

第一版只实现能够独立形成闭环的能力：

- 一个 CLI 入口；
- 一个 OpenAI-compatible Model Client；
- 一个清晰的 Agent Loop；
- Session 与概念上的 Turn/Step 执行边界；
- `read_file`、`list_files`、`search_files`、`write_file`、`edit_file`、`run_command`；
- 基础权限模式；
- Token 近似统计；
- 工具输出限制；
- JSONL Session 历史；
- 基础错误处理、取消和恢复；
- 单元测试和少量端到端测试。

第一版暂不实现：

- 复杂 PID/Kalman 控制器；
- 完整长期记忆；
- 多 Agent 图；
- 复杂 Goal/Todo/Quota 控制面；
- 插件市场或动态安装；
- 多模型自动路由；
- 完整 TUI；
- SQLite 历史索引。

## 12. 演进路线

### 阶段一：最小 Agent

- Model Client；
- Message 和 Model Event 类型；
- Tool Protocol；
- 基础工具；
- Agent Loop；
- CLI。

### 阶段二：可恢复 Runtime

- Session；
- JSONL 历史；
- 工具输出限制；
- 中断和恢复；
- 基础权限。

### 阶段三：上下文与 Skills

- Token 预算；
- 自动压缩；
- Skills 加载和注入；
- 项目级指令；
- 工具 Hook。

### 阶段四：记忆与评测

- 运行轨迹；
- Verifier；
- 简单文件型记忆；
- 任务后反思；
- 成功率、Token 和错误率评测。

### 阶段五：SubAgent 与长程控制

- 子 Session；
- 任务分解；
- 父子任务状态；
- Todo/Handoff；
- 基础执行预算或 quota。

## 13. 参考项目的取舍

### Pi

学习其简单、直接的模型—工具循环。epsilon 第一版的 Agent Loop 应保持类似的可读性。

### Codex

学习 `Session → Turn → Step` 分层、结构化历史、工具权限、上下文压缩、可恢复 Thread 和 Tool Registry/Runtime 分离。

不照搬其 Rust workspace、大量服务端协议、完整多 Agent 图和产品兼容代码。

### Hermes

学习 Python Runtime 的可读性、Skills 与工具的组合以及本地工作流的完整性。

### DeepSeek Harness

学习 Host、Plugin 和 Capability 的边界，让扩展能力可以装配到核心 Runtime 外部。但第一版不实现完整插件市场或动态安装系统。

### Bayesian-Agent

学习统一 Trajectory Evidence、Verifier 与 Runtime 分离、失败模式记录和后续 Skill 演化。它更适合作为后期评测/自进化层。

### LoopX

学习 Goal、Todo、Run History、Handoff 和长期状态之间的分离。LoopX 的长程控制面作为后期扩展，不进入第一版核心。

### MiniCode-Python

学习上下文压力、工具错误和成本都是可观测信号，以及工具结果截断和记忆预算控制。暂不照搬大量互相耦合的控制器。

## 14. 总结

epsilon 采用：

> 以 Session 和简单 AgentLoop 为核心的 Python Agent Runtime，将 Turn/Step 保持为未持久化的执行边界，使用 JSONL 保存会话历史，通过独立的 Context Manager 和 Tool Manager 管理上下文与工具，并为 Skills、Memory、Evaluation 和 SubAgent 预留清晰的扩展边界。

这套方案综合了 Pi 的简洁性、Codex 的结构化运行时、Hermes 的 Python 可读性、DeepSeek Harness 的扩展边界、Bayesian-Agent 的评测思想和 LoopX 的长期状态理念，同时避免第一版陷入 MiniCode-Python 式的过度工程化。
