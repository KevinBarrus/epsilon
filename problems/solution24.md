# 第二十四轮优化方案：修复容器共享工作区的所有权与补丁生成闭环

## 目标

让 SWE-bench Agent 容器中的测试命令、宿主文件工具和宿主补丁生成器能够安全操作同一工作区，并在补丁生成失败时正确标注评测阶段。生产模式不受影响。

## 唯一方案

评测容器仍使用官方实例镜像和 `/testbed`，但运行 Agent 命令时使用启动 Epsilon 评测进程的有效 UID/GID，而不是容器 root：

```text
宿主评测进程 UID:GID
  ↓
docker run --user <effective-uid>:<effective-gid>
  ↓
容器内 Python 测试写入 /testbed/__pycache__
  ↓
宿主工作区仍拥有这些文件
  ↓
宿主 create_patch() 可以删除缓存并生成补丁
```

这样保持了最重要的环境一致性：官方实例镜像、Python、依赖、工作目录和命令解释器均不变；只将绑定挂载上的文件写入身份对齐到宿主，避免 WSL/Docker 用户命名空间把容器 root 映射为 `nobody`。

不采用“容器结束后 chown 回宿主”“跳过无法删除的缓存”或“在容器与宿主之间反复复制工作区”：前两者会掩盖权限错误或依赖额外高权限操作，后者会破坏文件工具与命令工具的实时一致性。

## 代码改动顺序

### 问题一：让容器写入身份与宿主工作区对齐

1. 在 `evaluation/swebench_container.py` 中读取当前进程的有效 UID/GID
2. 启动任务容器时传入 Docker `--user <uid>:<gid>`
3. 保持官方实例镜像、`/testbed`、网络关闭、工作区挂载和容器清理逻辑不变
4. 不将 UID/GID 暴露给模型，也不将其加入生产工具层

### 问题二：精确归类补丁生成失败

1. 在 `run_task()` 中将 `create_patch()` 前的阶段标记为 `patch-generation`
2. 将运行时缓存删除、diff 生成或路径归一化异常记录为该阶段的评测环境错误
3. 保留已经完成的 Agent 轨迹、工具轮次、停止原因和 Agent 验证命令状态，避免异常结果丢失关键归因数据
4. 不伪造 `changed_files` 或 `official_harness_status`；未执行 Harness 时两者应明确为空

### 问题三：增加可复现测试

1. 单元测试断言容器启动命令使用当前有效 UID/GID、官方镜像、`/testbed` 挂载和禁网参数
2. 单元测试模拟补丁生成失败，断言结果标记为 `environment` / `patch-generation`，且保留已完成 Agent 的轨迹统计
3. 运行完整测试集

## 手动验收

先运行一个 Lite 实例的容器冒烟测试。任务结束后检查：

```bash
jq '{agent_execution_environment, agent_verification_status, official_harness_status, error_category, error_stage, changed_files}' \
  evaluation-results/<result-root>/results.jsonl
```

验收条件：

- 不出现 `PermissionError: ...pyc`
- `error_stage` 不再是错误的 `agent-loop`
- Agent 容器命令仍显示 `/testbed` 与官方镜像中的 Python 路径
- 流程能够进入官方 Harness，`official_harness_status` 为 `passed`、`failed` 或 `environment-error`，而非空值
- 宿主可删除新工作区中的 `__pycache__`

通过冒烟测试后，再重跑 retry7 的五个固定样本。仅在每个样本都有官方 Harness 结果后，才计算新的通过率并与 retry6 比较。

## 完成标记

- [x] 问题一：对齐容器与宿主的工作区写入身份
- [x] 问题二：补丁生成失败的阶段归因与轨迹保留
- [x] 问题三：单元测试与完整回归
- [ ] 验收：单实例冒烟与五个固定样本重跑
