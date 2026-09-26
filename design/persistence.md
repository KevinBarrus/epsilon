# Session + JSONL 持久化实现方案

## 1. 目标

在当前进程内 `Memory` 的基础上增加可恢复的会话：

- 每个会话拥有唯一的 `Session ID`；
- 每个会话对应一个 JSONL 文件；
- 用户消息和模型消息追加写入文件；
- 程序重启后可以根据 `Session ID` 恢复历史；
- 持久化逻辑不侵入 TUI 和模型客户端。

本阶段只持久化对话消息，不实现会话树、分支、工具事件、压缩摘要和长期记忆。

## 2. 存储位置

使用当前工作区根目录下的隐藏目录：

```text
.epsilon/
└── sessions/
    └── <session_id>.jsonl
```

选择工作区本地目录的原因：

- 不同项目的会话相互隔离；
- 会话可以随项目一起恢复；
- 不把运行时数据写入源码目录；
- 不依赖全局用户目录，便于面试现场展示和测试。

`.epsilon/` 属于运行时产物，实施时加入 `.gitignore`，不提交到代码仓库。

## 3. 数据格式

JSONL 不是一个大的 JSON 数组，而是“每行一个完整 JSON 对象”：

```jsonl
{"type":"message","role":"user","content":"你好"}
{"type":"message","role":"assistant","content":"你好，有什么可以帮你？"}
```

当前只允许 `user` 和 `assistant` 两种消息角色，与现有 `Message` 类型保持一致。文件名已经包含 `Session ID`，因此单条记录暂时不重复保存会话 ID 和时间戳。

采用追加式记录：不修改旧行，不重写整个文件。恢复时按文件顺序读取并重新构造 `Memory`。

该方案参考 Pi 的 append-only JSONL Session；暂不引入 Codex 的 SQLite 索引、Thread 图和复杂事件系统。

## 4. 模块职责

### `src/core/session.py`

定义 `Session`，负责一个会话的运行时状态：

- `session_id`；
- 当前 `Memory`；
- 当前工作区；
- 通过明确方法追加用户和模型消息。

`Session` 不负责 JSON 编码、文件路径拼接和 TUI 展示。

### `src/core/session_store.py`

定义 `SessionStore`，负责 JSONL 文件读写：

- 根据工作区创建 `.epsilon/sessions/`；
- 新建或追加指定 Session 的 JSONL 文件；
- 按文件顺序读取消息；
- 将 JSON 记录转换为 `Message`；
- 遇到无效 JSON、缺少字段或未知角色时抛出明确异常，不静默跳过损坏数据。

优先使用标准库 `pathlib`、`json` 和 `uuid`，不增加依赖。

## 5. 运行流程

### 新建会话

```text
启动应用
  ↓
生成 UUID 作为 Session ID
  ↓
创建 Session 和空 Memory
  ↓
等待用户输入
```

文件在第一次写入消息时创建，不为没有消息的会话提前创建空文件。

### 发送消息

```text
用户输入
  ↓
Session 追加 user 消息到 Memory
  ↓
SessionStore 追加 user JSON 记录
  ↓
使用 Memory 历史请求模型
  ↓
流式收集模型回复
  ↓
Session 追加 assistant 消息到 Memory
  ↓
SessionStore 追加 assistant JSON 记录
```

用户消息在请求模型前持久化，因此模型请求失败时，恢复后的会话仍能看到用户已经提交过的问题；错误提示不写入对话历史。

取消请求时沿用当前约定：已经生成的部分 assistant 回复同时写入 Memory 和 JSONL；“（已取消）”只显示在 TUI，不写入文件。

### 恢复会话

```text
指定 Session ID
  ↓
SessionStore 读取对应 JSONL
  ↓
按顺序恢复 Message
  ↓
构造带历史 Memory 的 Session
  ↓
继续对话
```

恢复只根据 JSONL 重建消息，不恢复活动请求、TUI 状态或模型连接。

## 6. 代码改动顺序

1. 新增 `SessionStore` 和 JSONL 记录读写；
2. 新增 `Session`，组合 `session_id`、工作区和 `Memory`；
3. 为新建、追加、读取和损坏记录添加单元测试；
4. 将 `ui.py` 的单个 `Memory` 替换为 `Session`；
5. 在应用启动时创建新 Session；
6. 暂时提供一个简单的恢复入口，先用明确的 `session_id` 参数，不提前增加会话选择界面；
7. 更新 `src/core/AGENTS.md`，说明 Session 与持久化模块边界；
8. 将 `.epsilon/` 加入 `.gitignore`。

实现过程中不修改 `screen.py` 的布局、滚动和输入处理逻辑。

## 7. 测试计划

新增 `tests/test_session_store.py`：

- 新建 Session 时生成合法且唯一的 ID；
- 第一次追加消息会创建目录和 JSONL 文件；
- 多条消息按追加顺序写入；
- 文件内容每行都是独立有效的 JSON；
- 从 JSONL 恢复后消息与原始历史一致；
- 不同 Session 写入不同文件；
- 无效 JSON、缺失字段和未知角色会抛出明确异常。

新增或更新 `tests/test_session.py`：

- Session 追加消息时同时更新 Memory 和持久化记录；
- 取消请求后的部分回复可以恢复；
- 模型错误提示不会进入历史。

继续运行现有完整测试集，确保 TUI 行为不受影响。

## 8. 验收标准

- 重启后可以根据 Session ID 恢复完整消息历史；
- JSONL 文件是一行一条记录，可以逐行读取；
- 用户消息、正常模型回复和取消后的部分回复都能恢复；
- TUI 不依赖文件格式和存储细节；
- 损坏记录不会被静默忽略；
- `.epsilon/` 不进入 Git；
- `uv run pytest` 全部通过。

## 9. 暂不实现的内容

- 会话列表和交互式选择器；
- 会话删除、重命名和分支；
- SQLite 索引；
- 压缩后的摘要记录；
- 工具调用和工具结果事件；
- 跨项目全局记忆；
- Codex 风格的异步长期记忆管线。

等基础持久化经过真实使用后，再根据恢复、文件增长和上下文长度问题决定下一步。
