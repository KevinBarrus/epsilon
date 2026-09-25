# 指令：并行 + worktree 隔离（照 oh-my-pi / ClaudeCode 形态）

## 目标

给多 Agent 补上"**文件隔离 + 并行**"能力，测一件事：**并行能不能把墙钟时间从 2.45× 压下来**。（诚实预期：token 仍会更高，因为文本通信的重读成本改不掉；本轮只赌"用更多 token 换更少时间"。）

## 前置：评测副本必须变成 git 仓库

现有评测副本**没有`.git`**（当初故意去掉）。worktree 依赖 git，所以复制后先：

```bash
git init && git add -A && git commit -m "baseline"
```

作为 worktree 的 base。非 git 仓库时**明确报错、不静默降级**（对标 oh-my-pi "requires a Git checkout"）。

## 组件 1：worktree 管理器（新增 `src/core/worktree.py`）

一个只负责 git 隔离的薄封装（照 oh-my-pi `worktree.ts` 的最小面）：

- `create_worktree(repo_root, task_id) -> path`：`git worktree add <同盘 sibling 目录> -b <branch>`，**branch 从当前 HEAD 拉**；
- `commit_worktree(path) -> branch`：在 worktree 内 `git add -A && git commit`（把 Worker 的新增/修改固化成一个提交）；
- `merge_branch(repo_root, branch) -> MergeResult`：在**全局仓库锁**下 `git cherry-pick`，冲突时 `cherry-pick --abort`、**保留分支、报告冲突**（不丢 Worker 的活）；
- `remove_worktree(repo_root, path)`：`git worktree remove`。

要点（照 oh-my-pi）：
- **所有 git 变更操作串行**（全局锁），因为并行 cherry-pick 会损坏工作区；
- **冲突不吞**：失败的分支保留，结果里列出"成功 / 冲突"清单。

## 组件 2：spawn_worker 集成

- 开启隔离时：
  1. 为该 Worker `create_worktree`；
  2. Worker 的 `workspace` **= worktree 路径**（它的 read/write/run_command 全在隔离目录里跑）；
  3. Worker 跑完 → `commit_worktree` → **在锁下** `merge_branch` 回主副本；
  4. `remove_worktree` 清理。
- **冲突处理**：该 Worker 的改动保留在分支上、报告冲突、不阻断其他 Worker。
- `execution_mode`：隔离开启时 = `parallel`（每个 Worker 有独立工作区，无写冲突）；关闭时维持 `sequential`。
- Worker 并发上限可配（建议 4），**合并步骤串行**。

## 组件 3：配置开关

- 新增 `isolation_enabled`（默认 **False**，保持现有行为不变）；
- 开启时走 worktree 隔离 + 并行；关闭时走现有的共享工作区 + 串行；
- 非 git 仓库 + 开启隔离 → 明确报错。

## 测试（先测后写）

1. worktree 往返：create → 写文件 → commit → merge → remove，主副本出现改动；
2. **冲突场景**：两个 Worker 从同一 baseline 改同一文件 → 一个 merge 成功、另一个报冲突且分支保留；
3. 并行：隔离下多个 Worker 真并发（时间重叠），合并串行；
4. 关闭隔离时行为与现在**完全一致**（回归不破坏）。

## 评测（对比三档）

同一任务、同 goal、同安全熔断、同模型 `deepseek-flash`：

| 档 | 工作区 | 执行 |
|---|---|---|
| 单 Agent（已有） | — | 33,317,726 token / 47:25 |
| 多 Agent 共享工作区（已有） | 共享 | 58,546,994 / 1:56:13 / 45-75 |
| **多 Agent worktree 并行**（本轮） | 隔离 | ? |

记录：总 token、总耗时、工具错误、**merge 冲突次数**、委派次数、各角色 token、结构完成度、完成真实性。

## 验收标准

1. 隔离下 Worker 真并行且互不干扰（各自的 worktree 独立）；
2. merge 冲突能报告、不丢改动；
3. 三档对比表齐全；
4. 关闭隔离时无回归（全量测试通过）；
5. 原仓库零改动。

## 约束

- 参考 oh-my-pi / ClaudeCode，**不照搬全量**（不做多后端 COW、不做 `.worktreeinclude`、不做周期 sweep），首版只做 git worktree 主路径；
- 先测后写；
- 跑前报费用确认；
- 诚实记录，冲突/失败如实标注。

---

转给执行 Agent。这轮之后，我们能第一次回答"**多 Agent 靠并行能不能把 2.45× 的墙钟时间压回来**"——如果压回来了，多 Agent 就是"花更多钱换更快"的取舍；如果压不回来，那"文本通信 + 重读"的成本就连并行也救不了。
