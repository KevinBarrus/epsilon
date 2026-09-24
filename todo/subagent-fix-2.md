# 指令：修复通信交接两处瑕疵 + 重新真机冒烟

## 瑕疵 1：父未转述 Worker 实际改动

**现状**：`SUBAGENT_PARENT_PROMPT` 要求"列出 Worker 的修改文件和验证命令"，模型照做但只传"目标行为"，没传"Worker 实际改了什么"。

**改法**：`src/core/subagent.py` 的 `SUBAGENT_PARENT_PROMPT` 最后两句改为要求转述**实际改动内容**：

> "委派 spawn_reviewer 时，必须在 context 里转述 Worker 的实际改动内容（改了哪个函数、从什么改成什么）和验证命令，让 Reviewer 聚焦检查这些改动；不要让它盲目扫描整个工作区。"

补测试：`tests/test_subagent.py` 断言 prompt 含"实际改动内容"或"从什么改成什么"。

## 瑕疵 2：工作区路径未显式注入

**现状**：父 Agent 不知道自己的工作区路径，冒烟里把临时工作区误写成项目根目录 `/home/kevinbarrus/projects/epsilon`，导致 Reviewer 的绝对路径读取落在工作区外被拒。

**改法**：

1. `src/core/context.py` 的 `ContextManager` 增加 `set_workspace_path(path: str)`，在 `_base_system_messages` 里追加一条 system 消息，内容形如：

   > "Current workspace root: `<path>`。所有文件工具的路径都相对这个根目录。"

2. `src/core/ui.py` 的 `run_chat` 创建 ContextManager 后调用 `context_manager.set_workspace_path(str(session_workspace))`；
3. `evaluation/subagent_workflow_smoke.py` 创建 ContextManager 后调用 `set_workspace_path(str(workspace))`；
4. 顺带检查 `evaluation/swebench.py` 的 `_context_builder` 是否也需要注入（评测工作区同样是临时快照），一并补上。

补测试：`tests/test_context.py` 断言 `set_workspace_path` 后 `_base_system_messages` 含工作区路径；`tests/test_subagent.py` 断言子 Agent 的 context_manager 也注入了工作区路径。

## 重新真机冒烟

修复后重跑：

```bash
cd /home/kevinbarrus/projects/epsilon
uv run python -m evaluation.subagent_workflow_smoke --confirm \
  --output evaluation-results/subagent-workflow-smoke.jsonl
```

跑前报费用确认。

## 验收标准（逐项报，不许只报"通过"）

1. 全量测试通过（当前 712 项）；
2. `reviewer_context` 原文里是否出现"实际改动"的表述（如"从 return a+b 改成 return a*b"之类），而不只是"应实现 a*b"；
3. `reviewer_read_paths` 是否**全部是正确路径**（相对路径或正确的临时工作区路径），不再出现 `/home/kevinbarrus/projects/epsilon` 这种误写；
4. 委派链仍为 `spawn_agent → spawn_worker → spawn_reviewer`、测试 OK、文件改对；
5. 各角色 token 无缺失。

## 约束

- 先补测试再改实现；
- 只改 prompt 和路径注入，不动三角色运行时、审批、并发逻辑。

---

转给执行 Agent。瑕疵 1 是改 prompt 一行的事；瑕疵 2 是"工作区路径显式注入"，落点在 ContextManager + 三处调用点，补测试覆盖。两项都做完、真机冒烟通过后回报，我严格验收。
