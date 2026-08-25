# 问题 24：容器写入共享工作区后导致补丁生成阶段权限失败

## 一、问题背景：为什么要做 retry7

第二十三轮优化将 SWE-bench 中的 `run_command` 从宿主机切换到官方实例镜像，目标是让模型看到的测试环境与最终官方 Harness 的裁判环境一致。

为了验收该链路，执行了五个此前失败的 Django Lite 样本：

```text
django__django-11019
django__django-11283
django__django-11905
django__django-13321
django__django-14155
```

运行结果目录为：

```text
evaluation-results/swebench-retry7/
```

命令最终打印 `swebench evaluation: 0/5 passed`。表面上这像是五个 Agent 任务都没有解决，但评测报告中的 `official_harness_status` 全部为空，需要先判断它们是否真正进入了官方裁判。

## 二、先理解这条评测流水线

当前一条真实任务的正常流程应为：

```text
准备宿主临时工作区
  ↓
启动官方实例镜像
  将宿主工作区挂载到容器 /testbed
  ↓
Agent 文件工具读写宿主工作区
Agent run_command 在容器 /testbed 运行测试
  ↓
Agent 结束后，在宿主工作区生成 Git diff 补丁
  ↓
独立官方 Harness 容器应用补丁并运行隐藏测试
  ↓
写入 official_harness_status
```

其中最后一步才可以判断补丁是否通过。只要流程在它之前异常退出，`0/5` 就不是模型解决率。

## 三、排查过程

### 1. 检查结果摘要，确认失败位置

先读取 `results.jsonl` 的以下字段：

```text
error_category
error_stage
error_message
agent_execution_environment
official_harness_status
changed_files
```

五个样本都具有相同特征：

```text
agent_execution_environment = official-instance-container
official_harness_status = null
changed_files = []
error_message = PermissionError: [Errno 13] Permission denied:
                '__init__.cpython-311.pyc'
```

这说明容器执行环境已经被启用，但 Agent 结束后未成功生成补丁，官方 Harness 从未执行。

### 2. 检查事件轨迹，确认 Agent 并没有在启动时失败

五个样本都存在 `agent_end` 事件、模型请求、工具调用和成功的 `edit_file` 事件：

| 实例 | 编辑次数 | Agent 停止原因 | 工具轮次 |
| --- | ---: | --- | ---: |
| 11019 | 2 | tool_limit | 40 |
| 11283 | 2 | completed | 38 |
| 11905 | 2 | tool_limit | 40 |
| 13321 | 1 | completed | 18 |
| 14155 | 2 | tool_limit | 40 |

因此，`changed_files=[]` 不表示模型没有改文件，而是后续 `create_patch()` 异常，来不及得到差异结果。

### 3. 检查容器环境是否真的生效

工具结果中出现了：

```text
/testbed/django/__init__.py
/opt/miniconda3/envs/testbed/bin/python
```

此前宿主 Python 3.12 与旧版 Django 的 `addDuration` 不兼容错误不再出现。由此可确认：问题 23 的“命令进入官方实例环境”已经生效，本问题不是回退到了宿主 Python。

### 4. 从异常栈定位具体代码路径

Agent Loop 返回后，`evaluation/swebench.py` 会调用：

```text
create_patch(...)
  ↓
_remove_runtime_artifacts(workspace)
  ↓
shutil.rmtree(workspace/**/__pycache__)
```

容器执行 Python 测试时在挂载目录生成了 `__pycache__/*.pyc`。随后宿主侧清理这些运行时文件时抛出 `PermissionError`，阻止了补丁生成和 Harness 验证。

### 5. 检查文件所有权，确认跨边界权限问题

对五个工作区中的缓存目录执行 `stat` 后，均得到类似结果：

```text
nobody:nogroup 755 .../django/__pycache__
nobody:nogroup 644 .../django/__pycache__/__init__.cpython-311.pyc
```

当前宿主用户不是 `nobody`，且目录权限是 `755`，因此宿主用户可以读取，却没有删除目录中内容的写权限。

容器以 `root` 用户运行；在当前 WSL/Docker 的用户命名空间映射下，容器 root 写入绑定挂载目录后，在宿主侧显示为 `nobody:nogroup`。这不是普通文件工具的权限问题，而是“容器用户”和“宿主工作区所有者”不一致。

## 四、问题定义

问题不是 SWE-bench 任务本身失败，也不是模型一定没有找到正确补丁，而是：

> Agent 容器以与宿主不同的用户身份写入共享工作区，产生了宿主无法删除的运行时文件；补丁生成阶段没有处理这种跨用户所有权，因此所有任务在官方 Harness 前失败。

这暴露出原方案遗漏的一项必要条件：共享工作区不仅需要路径一致，还需要容器与宿主对文件拥有可互操作的读、写、删除权限。

## 五、影响与数据解释

- retry7 的 `0/5` 不能作为通过率、失败率或简历指标
- 五个任务的官方补丁正确性均为“未验证”，不是“官方未通过”
- Agent 实际命令环境已改善，旧 Django 与宿主 Python 的不兼容证据已经消失
- 容器内禁网导致模型尝试 `pip install` 时 DNS 失败，会额外消耗轮次，但不是本次统一中断的直接原因
- 当前结果把补丁生成异常标为 `agent-loop`，阶段归因不准确，会妨碍后续分析

## 六、面试表达

可以这样解释这次失败：

> 我为了消除本地验证与官方裁判的环境差异，把 Agent 命令放进 SWE-bench 官方实例镜像执行，并把临时源码目录绑定挂载进去。第一次回归并没有直接看 0/5，而是先检查 Harness 状态，发现五条都没有进入官方裁判。继续沿着“Agent 结束 → 生成补丁”的链路排查，定位到容器 root 在 WSL 的绑定挂载上生成了宿主显示为 nobody 的 pyc 文件，宿主无法清理缓存，导致补丁生成失败。这个问题让我补齐了容器化评测中容易遗漏的一点：环境一致性不只是 Python 和依赖一致，文件所有权和清理语义也必须一致。

## 七、边界

- 不删除 retry7 目录，它是一次无效评测的排查证据
- 不把 retry7 的结果加入 baseline 或简历指标
- 不向模型开放 Docker 管理权限
- 不因这一问题开放容器网络，网络策略与文件所有权是两个独立议题
