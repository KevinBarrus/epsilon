# Epsilon

一个从零实现、可在终端直接运行的 Python Coding Agent。项目刻意保持克制：把模型调用、会话、上下文、工具、审批和 TUI 分层，便于阅读、演示和持续迭代。

## 能力

- OpenAI-compatible 模型的流式对话与推理过程展示
- 全屏 TUI：Markdown、代码高亮、长对话滚动、Slash Command、会话选择与工具审批
- JSONL 会话持久化；`epsilon resume` 可选择或按 ID 恢复会话
- 上下文预算、历史压缩和压缩失败的安全降级
- 本地文件、搜索、编辑和命令工具；写操作由用户审批，独立只读工具调用可并发执行
- 可选 stdio MCP Provider；Agent 只通过统一工具注册表调用工具
- Skills、项目 `AGENTS.md` 指令加载、写后验证提醒和结构化异常处理

## 架构

```text
TUI / Slash Commands
        │
        ▼
UI 编排 ── Session 持久化 ── Context Manager
        │                         │
        ▼                         ▼
                    Agent Loop ── Model Client
                        │
                        ▼
            Tool Registry / Permission / Executors
              ├── 本地文件与命令工具
              └── stdio MCP 工具
```

核心边界：`screen.py` 只渲染和收集交互；`ui.py` 负责编排；`agent_loop.py` 只驱动“模型 → 工具 → 模型”循环；工具权限与 TUI 确认界面相互隔离。

## 安装与配置

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/KevinBarrus/epsilon.git
cd epsilon
uv sync
uv run epsilon
```

首次启动会引导选择服务商、填写 API Key 和模型名称，并将配置写入 `~/.epsilon/settings.json`。也可以手动创建配置：

```json
{
  "model": {
    "base_url": "https://api.example.com/v1",
    "api_key": "your-api-key",
    "model_name": "your-model",
    "context_window": 100000,
    "reserve_tokens": 16000
  }
}
```

项目目录下的 `.epsilon/settings.json` 可按字段覆盖用户级配置。不要提交包含 API Key 的配置文件。

## 使用

```bash
uv run epsilon
uv run epsilon resume                 # 选择历史会话
uv run epsilon resume <session-id>    # 按 ID 恢复
```

- `Enter` 发送，`Ctrl+J` 换行，`Esc` 取消当前请求，`Ctrl+D` 退出
- 输入 `/` 打开命令列表；常用命令包括 `/model`、`/thinking`、`/compact`、`/status`、`/skills`、`/mcp`、`/export` 和 `/diff`
- 退出时会输出本次 Token 用量及恢复命令

写工具默认需要确认。审批界面支持 `Approve`、`Approve for this session` 和 `Reject`，避免模型在未确认的情况下修改文件或执行命令。

## 测试与评测

```bash
uv run pytest
```

除单元测试外，项目提供 SWE-bench Lite 的容器化真实任务评测。Agent 的命令工具在实例容器中执行，再由官方 Harness 裁判，避免“本地验证通过、官方环境失败”的验证失真。

最近一次固定样本实验使用 12 个 SWE-bench Lite Django 任务、80 轮工具预算、单次运行：**7/12 通过官方 Harness（58.3%）**。相同固定样本在 40 轮预算下为 5/12（41.7%），本次实验提升 **16.6 个百分点**。这是小样本的工程回归数据，不代表全量 SWE-bench 成绩。

评测入口与结果格式见 [`evaluation/`](evaluation/)，运行前需准备 SWE-bench 的 Python 环境和 Docker。

## 当前边界

Epsilon 是可运行的个人 Coding Agent，而非生产托管平台。当前评测样本有限；长任务的成本控制、更大规模基准集和更丰富的 MCP Provider 仍在持续迭代。
