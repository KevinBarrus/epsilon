# 第二十三轮优化方案：可替换命令执行环境与 SWE-bench 容器验证

## 目标

- 保持生产 Epsilon 默认在宿主执行命令
- 为命令工具预留可替换执行后端
- 仅在 SWE-bench 评测中将 `run_command` 路由到官方实例镜像
- 让模型修改的工作区与容器中执行测试的工作区保持同一份文件
- 保持官方 Harness 独立复核补丁，不将 Agent 自测当作最终结论

## 设计原则

```text
核心工具层：定义命令执行契约和默认宿主实现
        ↓
评测适配层：管理 SWE-bench 官方实例容器并实现该契约
        ↓
评测编排层：选择评测执行后端、收集轨迹、调用官方 Harness
```

- 核心 `tools/` 不依赖 Docker、SWE-bench 或评测数据集
- `evaluation/` 不修改生产启动链路，不增加普通用户的 Docker 前置条件
- 模型不能直接管理 Docker；容器创建、停止和删除只由评测编排层控制
- 不建立通用容器平台、镜像配置中心或多环境选择界面，避免超出当前需求

## 统一执行流程

```text
准备任务基线工作区
  ↓
从官方任务元数据取得实例镜像
  ↓
启动评测专属容器
  镜像：官方实例镜像
  容器工作目录：/testbed
  挂载：宿主评测工作区 → /testbed
  ↓
Agent 文件工具读写宿主评测工作区
Agent run_command 通过 docker exec 在 /testbed 执行
  ↓
模型读取容器内真实测试结果并继续决策
  ↓
停止并删除评测容器
  ↓
官方 Harness 使用同一实例镜像独立应用补丁并裁判
```

## 代码改动顺序

### 问题一：预留命令执行契约

1. 新增 `src/core/tools/command_executor.py`
2. 定义最小 `CommandExecutor` 协议：接收命令、工作目录、超时和取消信号，返回标准输出、错误输出与退出码
3. 将现有 `asyncio.create_subprocess_shell` 逻辑迁入 `HostCommandExecutor`
4. `create_run_command_tool()` 接收可选执行器，未传入时创建 `HostCommandExecutor`
5. 保持命令输出格式、超时文本、进程组清理和生产权限行为不变

唯一接口形态：

```python
class CommandExecutor(Protocol):
    async def execute(
        self,
        command: str,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandExecution: ...
```

`CommandExecution` 是只包含 `stdout`、`stderr`、`returncode` 的不可变数据对象。超时和取消仍由执行器明确处理，不把异常文本伪装成正常退出码。

### 问题二：实现 SWE-bench 容器生命周期

1. 新增 `evaluation/swebench_container.py`，只管理单个评测任务容器
2. 从 SWE-bench 官方任务元数据读取实例镜像，不拼接或猜测镜像名称
3. 使用与官方 Harness 一致的实例镜像、用户和容器工作目录 `/testbed`
4. 将 Agent 的临时评测工作区绑定挂载到 `/testbed`，使宿主文件工具写入立即对容器可见
5. 容器以常驻空命令启动；开始失败、执行异常、取消和评测结束均在 `finally` 中停止并删除
6. 容器操作失败转换为明确的评测环境错误，并写入结果轨迹

不复用官方 Harness 的“裁判容器”：Agent 容器和 Harness 容器必须隔离，避免 Agent 命令污染最终裁判状态；两者只复用同一官方实例镜像。

### 问题三：实现容器命令执行器

1. `SwebenchContainerExecutor` 实现 `CommandExecutor`
2. 每条命令通过 `docker exec -w /testbed <container> sh -lc <command>` 执行
3. 容器内设置硬超时；宿主侧再设置清理保护，避免 `docker exec` 遗留后台命令
4. 超时后终止对应容器进程；若无法可靠回收，销毁容器并将该任务标记为评测环境失败
5. 标准输出、错误输出和退出码回到既有 `run_command` 格式，模型无需知道命令来自容器

第一版关闭容器网络，不传递宿主 API Key、代理配置和用户目录。镜像依赖必须在启动前已构建完成。

### 问题四：接入 SWE-bench 编排与结果分类

1. 扩展 `SwebenchTask` 保留官方实例镜像标识，仅供评测运行时使用
2. 在 `run_task()` 中先创建容器执行器，再把它传入 `_tool_manager()`
3. `run_command` 的工具定义、权限和模型 Schema 保持不变
4. 在结果中记录 `agent_execution_environment=official-instance-container`
5. 保留已有的“写后本地命令状态”和“官方 Harness 状态”；前者改为“Agent 验证命令状态”，清楚标识运行位置
6. 官方 Harness 仍在 Agent 结束后独立执行，且仍是 `official_harness_status` 的唯一来源

### 问题五：加强失败后的受限验证收尾

1. 调整 `WriteVerificationPolicy`
2. 写后无命令时，保持现有的一次提醒
3. 写后命令失败且模型自然结束时，额外允许一次固定提醒：要求阅读失败输出、修复后重跑相关检查，或明确说明阻塞
4. 提醒总数最多一次；第二次自然结束、取消、工具上限、模型错误均直接结束
5. 不通过命令字符串猜测“是否为测试”，不要求无关任务强制运行测试

## 测试计划

- 未传执行器时，`run_command` 仍使用宿主执行器，现有命令、超时和取消测试全部通过
- 注入假的 `CommandExecutor` 后，工具将命令、工作目录、超时传递正确，并保留退出码语义
- 容器启动参数使用官方镜像和 `/testbed` 挂载，不依赖硬编码仓库名
- 容器启动失败、命令超时、用户取消、Harness 异常均会执行清理
- 容器输出可回传模型，文件工具修改在容器内可见
- 写后命令失败时最多追加一次提醒，不影响只读、普通对话和工具上限结束
- 评测 JSONL 与 HTML 同时记录 Agent 执行环境、Agent 验证命令状态和官方 Harness 状态
- 使用 retry6 中的 `11019`、`11283`、`13321` 重跑，比较宿主兼容性错误数量、验证命令结果与最终通过率

## 验收标准

- 生产启动不要求 Docker，普通 `run_command` 行为不变
- SWE-bench 中模型命令运行在官方实例镜像的 `/testbed`
- 文件工具修改能被容器命令立即读取，且 Agent 容器不会污染官方 Harness 容器
- 所有容器无论成功、失败、超时或取消均被回收
- 评测报告可明确区分 Agent 容器命令失败、Agent 验证不足、官方 Harness 未通过和官方 Harness 环境错误
- 失败后的验证提醒至多一次，不出现无限 Agent Loop

## 完成标记

- [x] 问题一：预留命令执行契约
- [x] 问题二：SWE-bench 容器生命周期
- [x] 问题三：容器命令执行器
- [x] 问题四：评测编排与结果分类
- [x] 问题五：失败后的受限验证收尾
- [ ] 验收：固定失败样本重跑与结果对比
