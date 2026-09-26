# 工具管理实现方案

## 一、总体方案

epsilon 采用“统一协议、分层来源、集中调度”的工具架构：

```text
Agent Loop
    ↓
ToolManager
    ↓
ToolRegistry
    ↓
RegisteredTool
    ├── ToolDefinition
    └── ToolExecutor
```

- Agent Loop 只负责模型与工具之间的循环
- ToolManager 负责参数校验、权限检查和执行流程
- ToolRegistry 负责统一查找、来源路由和工具名称冲突检查
- 本地工具与 MCP 工具可以分别发现和适配，最终注册到统一工具注册表
- 本地文件和命令工具在进入模型上下文前，按 UTF-8 16,000 字节或 400 行截断，并附带明确截断标记
- stdio MCP 工具结果暂未接入统一输出截断，作为后续增强单独处理
- TUI 只展示工具调用和结果摘要，避免占用过多空间

本方案采用 Pi 简单直接的模型—工具循环，参考 Codex 的工具定义与执行运行时分离，参考 Hermes 将 MCP 作为外部能力接入核心运行时

## 二、统一工具协议

新增最小数据结构：

- `ToolDefinition`：名称、描述、参数 Schema、工具来源、权限等级、幂等性
- `ToolRoute`：工具来源和提供者 ID，用于定位具体执行器
- `ToolCall`：调用 ID、工具名称、参数
- `ToolResult`：调用 ID、成功状态、结果文本、错误信息
- `ToolExecutor`：接收工具定义和 `ToolCall`，异步返回 `ToolResult` 的统一执行器
- `RegisteredTool`：工具定义与执行器的绑定关系

工具来源只有两种：`local` 和 `mcp`

工具路由由以下信息共同确定：

```text
工具来源 + provider_id + 工具名称
```

`source` 不能单独定位执行目标。多个 MCP Server 提供同名工具时，必须通过 `provider_id` 区分。模型可见的工具名称在统一注册表中必须全局唯一，第一版发现重名时直接拒绝注册

权限等级只有三种：`read`、`write`、`command`

幂等性用于描述重复执行同一个工具调用时，最终状态是否保持一致。第一版不要求所有工具都幂等，只保证可以安全保证的工具具备重复执行保护，并明确标记无法保证幂等的工具

扩展现有模型消息结构，使消息能够表示：

- assistant 工具调用
- tool 工具结果
- 工具调用 ID

模型返回的工具名称和参数不能直接执行，必须经过注册表查找、参数 Schema 校验和权限检查

## 三、本地工具与 MCP 工具

### 本地工具

第一版实现以下本地工具：

- `read_file`
- `list_files`
- `search_files`
- `write_file`
- `edit_file`
- `run_command`

每个本地工具只负责参数校验和具体执行，不负责注册、权限判断、TUI 展示和 Agent 循环

### MCP 工具

新增 `McpToolProvider` 协议，负责：

- 连接 MCP 服务
- 获取 MCP 工具定义
- 调用 MCP 工具
- 将 MCP 错误转换为统一 `ToolResult`

MCP 工具通过 `McpToolProvider` 发现和执行，通过 `McpToolRegistry` 暂存适配结果，启动时再注册到统一 `ToolRegistry`，不直接进入 Agent Loop

第一版只实现本地工具和 MCP 工具，不提前加入 HTTP、数据库等执行器。未来新增工具来源时，只需增加对应 `ToolExecutor` 和发现适配器，不修改 AgentLoop

## 四、ToolManager 与 Agent Loop

ToolRegistry 提供统一查找接口：

```python
register(registered_tool) -> None
get(name) -> RegisteredTool | None
definitions() -> list[ToolDefinition]
```

ToolManager 提供最小执行接口：

```python
list_definitions() -> list[ToolDefinition]
execute(tool_call) -> ToolResult
```

工具执行顺序固定为：

```text
查找工具
  ↓
校验参数
  ↓
权限检查
  ├── 自动允许或已获得本次会话授权 → 执行工具
  └── 需要用户确认 → 暂停 Agent Loop，等待 TUI 选择
        ↓
      执行工具或生成拒绝结果
  ↓
返回 ToolResult
```

Agent Loop 流程：

```text
发送用户消息
  ↓
请求模型
  ↓
普通文本 → 完成本轮
工具调用 → ToolManager 执行
  ↓
工具结果写入当前上下文
  ↓
再次请求模型
```

- 当前 Turn 内保存 assistant 工具调用和 tool 结果
- 工具执行失败也生成 tool 结果，让模型自行处理错误
- 模型最终回复完成后才写入 Session
- 用户取消时取消当前工具任务和后续模型请求

## 五、安全与并发

第一版采用中心化 `PermissionManager`，不在具体工具中重复实现权限逻辑

文件工具的默认绝对路径必须位于当前工作区内，禁止通过 `..` 越界

当前版本只允许文件工具访问当前工作区，暂不提供工作区外路径授权

权限规则：

- `read_file`、`list_files`、`search_files` 自动允许
- `write_file`、`edit_file` 需要用户确认
- `run_command` 每次调用都需要用户确认
- MCP 工具默认需要确认，只有明确标记为只读时才自动允许

权限检查分为两种路径：

- 自动允许路径：默认安全的只读工具，或当前 Session 已经授权的工具，直接执行，不暂停 Agent Loop，也不显示审批界面
- 用户确认路径：需要确认的工具暂停 Agent Loop，由应用层显示审批界面；用户完成选择后恢复执行流程

当前 Session 的授权只保存在内存中，授权键为“工具来源 + 工具名称”。退出 Session 后授权失效，不写入 JSONL

命令工具不参与 Session 授权。每次 `run_command` 都必须重新展示完整命令并获得单次确认

### TUI 工具审批

审批界面只替换现有输入框区域，不改变对话区和状态栏：

```text
正常状态：对话区 → 输入框 → 状态栏
审批状态：对话区 → 审批选项 → 状态栏
```

审批期间：

- 对话区保留已经生成的内容并暂停继续输入
- 输入框暂时隐藏
- 状态栏始终保留
- 当前选项使用蓝色，未选项使用默认字体颜色
- 用户使用上下箭头选择，按 Enter 确认
- Esc 等同于拒绝执行

写工具审批选项使用英文：

```text
Yes, proceed
Yes, and don't ask again for this tool in this session
No, and tell the model what to do instead
```

三种结果分别表示：

- `Yes, proceed`：只允许当前工具调用
- `Yes, and don't ask again for this tool in this session`：记录当前 Session 授权，后续相同工具无需再次询问
- `No, and tell the model what to do instead`：拒绝调用，将拒绝结果交给模型，并恢复输入框供用户继续发送消息

命令工具只显示 `Yes, proceed` 和 `No, and tell the model what to do instead`，不提供会话级授权

无论用户选择哪一项，按下 Enter 后都必须恢复原有 TUI 布局：对话区、输入框和状态栏恢复正常，不在对话区追加审批说明

审批组件只负责选项展示和返回审批结果，不负责工具执行和权限判断。`PermissionManager` 不依赖 `screen.py`，应用层通过 `ApprovalHandler` 注入审批回调。审批界面应作为输入区域的替代内容接入现有布局，不能修改对话滚动、流式输出和输入框编辑逻辑

第一版不实现操作系统级沙箱，文件工具的安全边界由工作区路径检查和确认回调组成

`run_command` 使用工作区作为当前目录，但 shell 命令本身可能访问工作区之外的位置，因此不能仅依赖文件路径校验。当前实现不是沙箱：它只保证每条命令在执行前完整可见、每次都需要单次确认。网络、环境变量、CPU、内存和文件系统隔离留待后续实现

第一版所有工具调用按模型返回顺序串行执行：

- 避免写操作竞态
- 保证工具结果顺序稳定
- 兼容不支持并发的 MCP 服务
- 简化取消和 TUI 展示

后续确认只读工具安全后，再增加只读工具并行执行

### 工具幂等性

第一版采用“工具声明幂等性，ToolManager 负责执行约束”的方案：

- `read_file`、`list_files`、`search_files`：只读，天然幂等
- `write_file`：写入目标最终内容；如果当前内容已经与目标内容一致，直接返回成功
- `edit_file`：按旧内容子串定位并替换为新内容，保留未匹配到的其余内容（含末尾换行）；旧内容已不存在且新内容已存在时返回成功，两者都不存在时拒绝执行
- `run_command`：默认标记为非幂等，每次调用需要确认，禁止自动重试
- MCP 工具：由工具定义声明幂等性；未声明时按非幂等处理并需要确认

串行执行只能避免并发竞态，不能自动保证幂等性。幂等性必须由具体工具的执行语义、状态校验和重复调用保护共同保证

## 六、TUI 展示

工具调用作为独立的简短活动条目展示：

```text
▸ read_file  src/core/model.py
  ✓ 已读取 42 行
```

展示规则：

- 开始时显示工具名和关键参数摘要
- 完成时显示成功或失败状态
- 结果只显示单行摘要
- 长参数和换行内容使用省略号
- 完整结果不展示在对话区，但完整传给模型
- 工具失败显示简短错误摘要

TUI 只接收工具事件，不依赖具体本地工具或 MCP 工具实现

确认流程由应用层提供 `ApprovalHandler` 回调，`PermissionManager` 不直接依赖 `screen.py`

## 七、代码改动顺序

1. 扩展模型消息和工具调用数据结构
2. 新增统一工具协议、`ToolManager` 和 `LocalToolRegistry`
3. 实现 `read_file`、`list_files`、`search_files`
4. 实现 `write_file`、`edit_file`、`run_command`
5. 为本地工具增加幂等性声明和重复执行保护
6. 新增 `PermissionManager`、确认回调和额外授权路径边界
7. 扩展模型客户端，解析文本响应和工具调用响应
8. 实现 Agent Loop，调用 `ToolManager` 并将完整工具结果写回模型上下文
9. 接入应用层审批回调：默认安全工具和已获 Session 授权的工具直接执行；需要确认的工具才暂停 Agent Loop
10. 新增输入区域审批选择组件，只替换输入框区域并保留状态栏，不修改既有对话展示逻辑
11. 接入工具事件的 TUI 摘要展示
12. 新增 `McpToolProvider` 和 `McpToolRegistry` 接口
13. 补充 `provider_id`、`ToolRoute` 和统一 `ToolRegistry`，完成全局工具名称冲突检查
14. 实现最小 stdio MCP Provider，并将发现的工具接入统一注册表
15. 更新 Session JSONL，使工具调用和工具结果可以恢复
16. 更新 `src/core/AGENTS.md`
17. 运行完整测试集

## 八、测试与验收标准

测试至少覆盖：

- 工具注册、查找和重复注册
- 参数校验失败
- 工作区路径越界
- 额外路径未授权时被拒绝，授权后按读写范围执行
- 只读工具自动通过权限检查
- 写操作和命令执行需要确认
- 默认安全工具不会暂停 Agent Loop
- 已获得当前 Session 授权的工具不会重复弹出审批界面
- 审批界面只替换输入框区域，状态栏始终保留
- 审批选项支持上下箭头、蓝色选中项和 Enter 确认
- 审批完成后对话区、输入框和状态栏恢复原状
- 拒绝工具调用后，拒绝结果能够交给模型
- 幂等工具重复调用后的结果保持一致
- `write_file` 内容相同时不会重复写入
- `edit_file` 已完成、状态冲突和正常修改三种情况
- 多个工具来源通过 `provider_id` 正确路由
- 同名工具在统一注册表中被拒绝
- 非幂等命令不会被自动重试
- 未声明幂等性的 MCP 工具按非幂等工具处理
- 工具执行成功、失败和取消
- 多个工具按顺序执行
- 模型连续请求工具并最终生成回复
- 模型请求未注册工具时被运行时拒绝
- 工具参数不符合 Schema 时被运行时拒绝
- MCP 工具转换为统一工具定义和结果
- TUI 显示工具调用、成功和失败摘要
- Session 恢复工具调用和工具结果

验收标准：

- 模型可以调用本地工具并根据结果继续回答
- 本地工具和 MCP 工具没有互相依赖
- 工具执行逻辑不进入 `screen.py`
- 未经确认的写操作和命令不会执行
- 未注册工具和非法参数不会进入 handler
- 本地文件和命令工具的结果在既有输出预算内进入模型上下文
- TUI 能证明工具确实执行，同时不会被完整输出占满
- `uv run pytest` 全部通过

## 九、明确暂不实现

- stdio MCP 工具输出截断
- 工具结果摘要写回模型
- 只读工具并行执行
- 复杂沙箱
- 命令工具的完整沙箱和网络权限细分
- 通用工具重试策略，第一版只禁止非幂等工具自动重试
- 动态工具发现
- HTTP、数据库等未使用的工具执行器
- 复杂的工具能力分类和通用元数据体系
- 工具循环检测
- 多 Agent 工具隔离
