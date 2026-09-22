# 问题 29：长任务评测设计（同一 Session 多任务，量化驱逐净收益）

> 本文档只做设计，不含实现。经审阅后再进入编码。

## 一、背景

主表 B/C（12 题、80 轮、单 bug 短任务）已跑完，结论写在 `problems/main-table-report.md`：

- 防火墙对 token 的真实收益只有 **2–3%**，远低于 15% 的 go/no-go 线；
- 驱逐在默认预算下**全程零触发**：阈值是 `compaction_threshold // 2 = (100000 - 16000) // 2 = 42000` token，而单个 bug 任务的单次请求上下文始终低于它；
- 单题冒烟里看到的"省一半"是运行方差（轮次数差异），不是防火墙本身。

这说明：**驱逐/压缩的主场不是单 bug 短任务，而是"上下文自然积累"的长任务**。单 bug 任务做完即结束，上下文没有机会长到阈值以上；长任务里，历史工具输出会一轮轮重复计入输入，驱逐才有发挥空间。

## 二、目标

回答一个问题：

> 真实长任务里，当驱逐"偶尔触发"时，净收益是正还是负？
> 即：**"少重发旧工具输出"省下的 token，能否盖过"改写历史导致前缀缓存失效"的损失，以及"占位符取回"带来的额外轮次**。

这是一个方向性问题，不做统计显著性结论，只做单样本的工程对照。

## 三、实验设计

### 3.1 起点

- 任务：`django__django-11001`，base commit `ef082ebb84f00e38af4e8880d04e8365c2766d34`；
- 一个 workspace + **一个 Session**，整个实验只创建一次，中途不重置、不新建会话；
- 使用官方实例镜像 `swebench/sweb.eval.x86_64.django_1776_django-11001:latest`（本地已缓存）。

### 3.2 连续任务序列（同一 Session 内依次下发）

| 阶段 | 内容 | 上下文关系 |
|---|---|---|
| T1 | 修复 11001 的 issue（`OrderBy` 对多行 `RawSQL` 的错误去重） | 建立初始上下文 |
| T2 | 为 T1 改动的功能补回归测试，加到 `tests/ordering/tests.py` | 继承 T1 全部历史（含 T1 的读文件、跑测试、报错与修复过程） |
| T3 | 重构 T1 的修复：把"剥离 ORDER BY 方向 + 计算去重键"提取为 `SQLCompiler` 私有方法，行为不变 | 继承 T1+T2 全部历史 |

关键约束：三个任务共用同一个 `Session`、同一个 workspace、同一个 `ArtifactStore`；每阶段结束**不压缩、不清空、不重建**，只追加一条新的用户消息。

### 3.3 两档对照

| 档位 | firewall | eviction | 驱逐阈值 |
|---|---|---|---|
| E0（对照） | 开 | **关** | —（不驱逐） |
| E1（实验） | 开 | **开** | **20000 token** |

- firewall 两档都开：本实验只隔离驱逐这一个变量，防火墙收益已在主表证明很小且稳定；
- 压缩阈值两档都保持默认（`compaction_threshold = 84000`），不额外调低；E1 由于驱逐把上下文压在 20k 之上不远，预计**永远不触发压缩**，E0 可能在 T2/T3 触发一次压缩——这正是"驱逐替代压缩"的价值体现，属于被测对象；
- 阈值 20000 的选择理由：默认 42000 在长任务里也要到中后段才可能触发，20k 保证 T1 中后段就开始出现驱逐批次，样本内能观察到多次"触发—前缀失效—继续积累"的完整周期。

## 四、验证方式

每个阶段的"完成"必须有可执行判据，不能靠人工感觉。

### 4.1 T1：硬验证（官方 Harness）

- 复用 `evaluation/swebench.py::create_patch(baseline, workspace)` 生成补丁；
- 复用 `evaluation/swebench.py::verify_patch(task, patch, result_root/"harness", harness_python)` 跑官方 Harness；
- 判定：`verification.passed is True`（即 `resolved`）。
- 这是唯一有 ground truth 的判据；base_commit 与实例一致，可信。

### 4.2 T2：软验证（两条都要满足）

1. **新增测试可运行且通过**：在容器内执行固定命令
   ```
   /opt/miniconda3/envs/testbed/bin/python tests/runtests.py ordering --verbosity 2
   ```
   判定：`exit_code == 0`。
2. **修复未被破坏**：对 T1+T2 后的 workspace 再跑一次官方 Harness，判定 `resolved == True`。
3. **改动范围符合承诺**：`create_patch` 出来的 `changed_files` 只允许出现在 `tests/ordering/` 下（T2 不允许改生产代码）。

> 第 1 条才是"新增测试"的判据；第 2 条只保证原 bug 修复仍有效。两条都要，缺一不算完成。

### 4.3 T3：软验证（三条都要满足）

1. **相关测试仍通过**：同一命令 `tests/runtests.py ordering`，`exit_code == 0`；
2. **修复未被破坏**：官方 Harness 再跑一次，`resolved == True`；
3. **重构形态可判定**：`changed_files` 只允许出现
   - `django/db/models/sql/compiler.py`（生产代码），
   - `tests/ordering/`（如 T2 已有改动）；
   且补丁中**不得出现文件删除**（`+++ /dev/null`）。

### 4.4 关于"软验证不够硬"的诚实标注

- T2/T3 没有官方 ground truth，"测试通过"只能证明没破坏既有行为，不能证明重构更优或测试覆盖充分；
- 所有 T2/T3 结果一律标注为 `soft-pass`，**不得计入 12 题 SWE-bench 通过率**；
- T1 的 Harness 结果可单独作为 `hard-pass` 呈现。

### 4.5 关于官方 Harness 与测试文件的冲突（实现前必须预检）

SWE-bench 官方 Harness 在验证时会应用实例自带的 test patch。若我们在 T2 中修改了同一批测试文件，test patch 可能应用失败或覆盖我们的新增测试。因此：

- T2/T3 的"新增测试"判定**只用容器内命令**（4.2 第 1 条、4.3 第 1 条），不依赖 Harness；
- 若某阶段 Harness 报环境失败且日志显示 test patch 冲突，该阶段 Harness 结果标为 `environment`，不影响该阶段 soft-pass 判定，但要在报告中如实列出。

## 五、指标

| 指标 | 采集方式 | 用途 |
|---|---|---|
| `actual_tokens`（分阶段 + 累计） | `TimedModelClient` 的 usage，按阶段前后差分 | 主指标：E1 vs E0 |
| `cache_hit_rate`（分阶段 + 累计） | `TimedModelClient.total_cached_tokens / total_actual_tokens` | 抵消项：驱逐打断前缀缓存的代价 |
| 驱逐次数 | `_context_builder` 追加的 `{"type": "eviction"}` 事件计数 | 确认 E1 真的触发 |
| 压缩次数 | `{"type": "compaction"}` 事件计数 | 观察驱逐是否替代了压缩 |
| 每任务完成度 | T1 `hard-pass`；T2/T3 `soft-pass` | 保证不是"省 token 但做不完" |
| 总工具轮次 | `AgentRunResult.tool_rounds` 累加 | 观察取回开销是否增加轮次 |
| `read_artifact` / `artifact://` 取回次数 | 工具调用事件中 `path` 含 `artifact://` 的数量 | 直接量化"取回开销" |

### go/no-go 判据（本实验专用）

- **净收益为正**：E1 的累计 `actual_tokens` ≤ E0 的 85%（即降幅 ≥ 15%），且 T1 hard-pass、T2/T3 soft-pass 全部通过；
- **净收益为负**：E1 累计 token 高于或与 E0 持平（如降幅 < 15%），即使驱逐触发很多次；
- 报告必须同时给出 `cache_hit_rate` 的下降幅度，说明 net token 的构成。

## 六、实现点（文件 / 函数级）

### 6.1 新入口 `evaluation/long_task.py`

复用而非复制：

- 复用 `swebench.py::prepare_repository`、`swebench_workspace.py::prepare_evaluation_workspace` 准备 workspace 与 `session_root`；
- 复用 `swebench_container.py::SwebenchTaskContainer` / `SwebenchContainerExecutor`；
- 复用 `swebench.py::_tool_manager`（含 `artifact_store` 注入）；
- 复用 `swebench.py::_context_builder`（扩展后，见 6.2）；
- 复用 `swebench.py::create_patch` / `verify_patch` / `HarnessResult`；
- 复用 `evaluation/online.py::TimedModelClient`。

新增结构：

```python
@dataclass(frozen=True)
class LongTaskStage:
    name: str                 # "T1" / "T2" / "T3"
    prompt: str               # 阶段指令
    validation: str           # "harness" | "ordering-tests"
    allowed_changes: tuple[str, ...]   # 允许改动的路径前缀

@dataclass(frozen=True)
class LongTaskSpec:
    instance_id: str
    source: str
    stages: tuple[LongTaskStage, ...]

async def run_long_task(
    spec, result_root, harness_python, *,
    eviction_enabled: bool,
    eviction_threshold_tokens: int | None,
    thinking: str = "high",
    firewall_enabled: bool = True,
    max_tool_rounds_per_stage: int = 80,
) -> LongTaskResult
```

`run_long_task` 主流程（顺序固定）：

1. `prepare_repository` → `prepare_evaluation_workspace` ×2（baseline / prepared）；
2. `SwebenchTaskContainer(prepared.workspace).running()`；
3. 创建 `ArtifactStore.for_workspace(prepared.session_root)`、`Session(prepared.session_root)`、`TimedModelClient`、`_tool_manager`，`artifact_store.set_session_id(session.session_id)`；
4. `ContextManager` 经 `_context_builder` 创建，传入 `eviction_enabled` 与 `eviction_threshold_tokens`；
5. **单次 `AgentLoop` 实例**贯穿三个阶段（同一 Session），或每阶段新建但共享同一 Session——两者语义等价，优先单实例；
6. 对每个 stage：
   - `session.add_user_message(stage.prompt)`；
   - 记录阶段前 `TimedModelClient` 计数器基线（请求数、token、驱逐/压缩事件数、工具轮次）；
   - `await agent.run(session.get_messages(), on_event=..., build_context=...)`；
   - `session.add_message(m)` 持久化 `agent_result.new_messages`；
   - 按 `stage.validation` 执行验证（见 6.3）；
   - 计算阶段差分，写一条 `long_task_stage` 记录；
7. 全部阶段结束后 `session.flush_persistence()`、`session.close()`；
8. 写 arm 级汇总。

CLI：

```
.venv-swebench/bin/python -m evaluation.long_task --confirm \
  --instance-id django__django-11001 \
  --result-root evaluation-results/long-task-E1 \
  --eviction --eviction-threshold 20000 \
  --harness-python .venv-swebench/bin/python \
  --max-tool-rounds 80
```

### 6.2 驱逐阈值可配置

现状：`src/core/context.py::ContextManager._maybe_evict` 里阈值硬编码

```python
threshold = self._budget.compaction_threshold // 2
```

改动点（全部为参数透传，不改驱逐算法本体）：

1. `ContextManager.__init__` 增加 `eviction_threshold_tokens: int | None = None`；
2. `_maybe_evict` 改为：
   ```python
   threshold = (
       self._eviction_threshold_tokens
       if self._eviction_threshold_tokens is not None
       else self._budget.compaction_threshold // 2
   )
   ```
   `KEEP_RECENT_TOOL_OUTPUTS`、`_apply_evictions`、`_eviction_placeholder` 不动；
3. `src/core/config.py`：`Settings` 增加 `eviction_threshold_tokens: int | None = None`，从 `model.eviction_threshold_tokens` 解析，非正整数报 `ConfigError`；
4. `src/core/ui.py::run_chat`：创建 `ContextManager` 时传 `eviction_threshold_tokens=settings.eviction_threshold_tokens`；
5. `evaluation/swebench.py::_context_builder`：增加同名参数并传给 `ContextManager`；
6. 测试：`tests/test_context.py` 补"自定义阈值生效 / 不传时退回默认的一半"；`tests/test_config.py` 补解析与非法值。

### 6.3 逐任务验证与落盘

新增 `evaluation/long_task_validation.py`（或在 `long_task.py` 内，保持单文件职责清晰优先独立文件）：

```python
async def run_ordering_tests(executor, workspace, timeout_seconds) -> StageValidation:
    """在容器内执行固定测试命令，返回 exit_code 与尾部输出。"""

async def run_harness_check(spec, baseline, workspace, result_root, harness_python) -> StageValidation:
    """生成补丁并跑官方 Harness，返回 resolved 与环境失败原因。"""
```

- `run_ordering_tests` 通过 `SwebenchContainerExecutor.execute(command, workspace, timeout)` 执行；
- `run_harness_check` 通过 `create_patch` + `verify_patch`；
- 命令与允许改动范围**不允许写在三个 stage 里各写一遍**，集中在 `LongTaskSpec` 构造处定义一个常量：
  ```python
  ORDERING_TESTS_COMMAND = (
      "/opt/miniconda3/envs/testbed/bin/python tests/runtests.py ordering --verbosity 2"
  )
  ```
- 该命令在实验前必须做一次**预检**（不调模型）：手工在容器里对 T1 完成后的代码执行一次，确认 exit 0；预检结果写进报告，命令不通就先修命令再开跑。

落盘 schema（每阶段一行，写 `result_root/long_task.jsonl`，复用 `evaluation/storage.py::append_result` 的 JSONL 风格或独立写入）：

```json
{
  "type": "long_task_stage",
  "run_id": "<uuid>",
  "arm": "eviction_on",
  "eviction_threshold_tokens": 20000,
  "stage": "T1",
  "passed": true,
  "validation_kind": "harness",
  "command": null,
  "exit_code": null,
  "detail": "resolved",
  "changed_files": ["django/db/models/sql/compiler.py"],
  "tool_rounds": 43,
  "model_requests": 44,
  "actual_tokens": 1234567,
  "cached_tokens": 1100000,
  "cache_hit_rate": 0.891,
  "eviction_events": 6,
  "compactions": 0,
  "read_artifact_calls": 2,
  "duration_ms": 123456
}
```

arm 级汇总行：

```json
{
  "type": "long_task_arm",
  "arm": "eviction_on",
  "stages": ["T1", "T2", "T3"],
  "hard_pass": true,
  "soft_pass": [true, true],
  "total_actual_tokens": 3456789,
  "total_cached_tokens": 3000000,
  "total_eviction_events": 14,
  "total_compactions": 0,
  "total_tool_rounds": 120,
  "total_read_artifact_calls": 5
}
```

### 6.4 落地顺序（每步先补测试再实现）

1. `ContextManager` 阈值参数 + 测试；
2. `Settings.eviction_threshold_tokens` + 测试；
3. `_context_builder` 透传 + 测试；
4. `long_task_validation.py` 的两个验证函数 + 单测（用假 executor / 假 verify_patch）；
5. `long_task.py` 主流程 + 单测（monkeypatch 容器与 client，沿用 `test_evaluation_swebench.py` 的 Fake 模式）；
6. CLI 与端到端预检（不调模型）。

## 七、成本与风险

### 7.1 开发量

| 项 | 估计 |
|---|---|
| 阈值参数透传（context/config/ui/swebench + 测试） | 0.5 天 |
| `long_task.py` 主流程 + 假实现单测 | 1 天 |
| 验证模块 + 预检 | 0.5 天 |
| 合计 | 约 2 天 |

### 7.2 运行成本

- 每个 arm：1 个 Session × 3 阶段，上下文持续增长，预计单 arm `actual_tokens` 在 300 万–800 万量级；
- 2 个 arm 合计约为主表单题的 2–4 倍 API 费用；
- 另外 6 次官方 Harness（每阶段一次 × 2 arm，每次约 25 秒 Docker 时间，不花钱）；
- 墙钟时间：预计 3–5 小时（可后台跑、分阶段轮询）。

### 7.3 风险与如实标注

1. **软验证不够硬**：T2/T3 只能证明"没破坏"，不能证明质量；报告一律标 `soft-pass`，禁止并入 SWE-bench 通过率；
2. **单样本无统计力**：每个 arm 只跑 1 次、1 个任务，结论只能定性，不能外推；报告须写明"单样本工程观察"；
3. **官方 Harness 与测试文件冲突**：T2 改 `tests/ordering/tests.py` 可能与实例 test patch 冲突，按 4.5 处理并如实记录；
4. **固定命令依赖容器环境**：`/opt/miniconda3/envs/testbed/bin/python` 与 `tests/runtests.py ordering` 需预检确认；容器内无网络（`--network none`），不能装包；
5. **驱逐阈值 20000 可能过度触发**：若 E1 每几千 token 就驱逐一次，前缀缓存会被反复打断，正是要观察的失败模式；若触发过密导致任务做不完，则记录为"驱逐过度"的负结果，而不是调阈值重跑到好看；
6. **上下文增长不可控**：模型在 T1 可能反复读大文件，导致 E0 提前压缩、E1 提前驱逐，两 arm 的上下文形态差异会放大；报告需给出每阶段请求数与平均 prompt token，便于判断可比性。

## 八、交付物

1. 本文档（`problems/problem29.md`）；
2. 实现完成后的长任务入口与测试；
3. 实验报告：E0 / E1 两 arm 的逐阶段表、指标对比、go/no-go 判定、以及 7.3 中各风险的实际情况记录。
