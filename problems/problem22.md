# 问题 22：项目上下文与修改后验证未形成 Harness 闭环

## 问题一：Epsilon 未加载工作区项目指令

生产运行时只加载内置 `src/core/prompts/agent.md`，不会发现并注入工作区中的 `AGENTS.md`。模型因此看不到项目特定的架构边界、常用命令、测试要求和协作约束。

同一模型在不同 Coding Agent 中收到的系统提示词、项目说明、工具契约和当前目录事实不同，不能将模型名相同等同于实际 Agent 能力相同。

## 问题二：修改后的自然结束没有验证证据约束

`AgentLoop` 在模型不再请求工具时直接结束。发生 `write_file` 或 `edit_file` 后，模型可以不运行验证命令，或将无关成功命令当作验证，自称完成后结束。

这不是模型流式、工具并行或工具回合数问题，而是 Harness 没有在结束点检查“修改—验证”闭环。

## 问题三：SWE-bench 仓库通常没有项目指令文件

retry5 的 Django 工作区没有 `AGENTS.md`，但包含 `README.rst`、`CONTRIBUTING.rst`、`tox.ini`、`setup.cfg` 和 `tests/README.rst`。因此，仅实现项目指令加载不会直接改善此类真实评测；模型仍需主动获得仓库测试和运行方式的线索。

## 问题四：SWE-bench Agent 验证环境与官方 Harness 不一致

Agent 在宿主 Python 3.12 中运行旧 Django 快照测试，出现 `TestResult has no addDuration method` 等兼容性错误；官方 Harness 在任务容器中验证补丁。Agent 会收到失真的本地测试反馈，最终官方结果又不会回传给 Agent 修复。
