# 指令：查并发 + 接入 Writer/Reviewer 子 Agent

## 第一部分：查并发（纯排查，不改生产逻辑）

**背景**：冒烟里 4 个 Scout 并行时耗时 5.2 倍，理论上真并行应接近"最慢那个"。需确认 4 个 Scout 的模型请求到底有没有真并发。

**做法**：

1. 在 `evaluation/subagent_smoke.py` 或 `src/core/subagent.py` 的 `_run_scout` 里，给每个 Scout 的模型请求打时间戳（请求开始/结束），或记录每个 Scout 的起止时间；
2. 跑一次"多独立调查"冒烟（4 个 Scout 并行，真机），导出时间线；
3. 判断：
   - 4 个 Scout 的请求时间**重叠** → 真并发，5.2 倍耗时是"每个 Scout 本来就慢"，另找原因；
   - **不重叠（串行排队）** → 找串行根因：shared client 是否串行化、DeepSeek 是否 rate limit、asyncio 是否真并发。

**产出**：时间线结论 + 根因判断，先报告，**不要急着改**（等确认根因再定修法）。

## 第二部分：接入 Writer 和 Reviewer（按 problem30 第二节）

**背景**：当前只有只读 Scout（spawn_agent）。要做重构类任务，必须有写（Worker）和审（Reviewer）子 Agent。

**做法**：在 `src/core/subagent.py` 增加两个角色专用工具（不是给 spawn_agent 加 role 参数，因为 execution_mode 固定在工具定义上）：

### 1. `spawn_worker(task)`

- 工具集：`read_file` / `list_files` / `search_files` / `write_file` / `edit_file` / `run_command`（全部本地工具）；
- `execution_mode = sequential`（写操作串行，避免共享工作区冲突）；
- **写操作和命令必须走审批**——复用父 Agent 的 `PermissionManager` / 审批回调，不因委派绕过权限；
- 深度限制：Worker 工具集里没有 `spawn_*`，不能创建孙 Agent；
- 独立上下文，只回有界摘要；摘要必须列出修改文件、验证命令和验证结果。

### 2. `spawn_reviewer(task)`

- 工具集：`read_file` / `list_files` / `search_files` / `run_command`（只读 + 命令）；
- `execution_mode = sequential`；
- 系统提示词要求它只运行验证相关命令，但**每次命令仍需审批**（它不是安全沙箱）；
- 深度限制同上；
- 摘要必须给出通过项、发现的问题和证据。

### 3. 复用与共享

- 复用现有的 `_run_scout` 骨架（独立 `ToolManager` / `ContextManager` / `AgentLoop`、超时、轮次上限、摘要上限），把角色差异抽成参数（工具集、execution_mode、系统提示词、是否需要审批）；
- 三个角色都继承主模型和思考强度、独立上下文、深度限制 = 1。

### 4. `/subagent` 命令与开关

- `/subagent` 命令的展示里加入 Worker、Reviewer 的 on/off 状态（三个角色可独立开关，或统一开关，实现上优先"统一开关 + 展示三角色状态"）；
- 系统提示词注入三角色的使用说明。

## 验收标准

- 全量测试通过（当前 703 项）；
- 单测覆盖：spawn_worker 有写工具且 sequential、spawn_reviewer 有命令且 sequential、两个角色都不能 spawn 孙 Agent、Worker 写操作走审批、三角色独立上下文；
- 冒烟（真机，跑前报费用）：一个"定位 → 修改 → 验证"的小任务走通 Scout → Worker → Reviewer 委派链，记录 token / 耗时 / 委派次数。

## 约束

- 先补测试再改实现；
- 不动 Scout 的现有逻辑，只新增 Worker/Reviewer 并把公共骨架抽出来；
- 不改生产上下文管理、不改会话持久化协议。

---

转给执行 Agent。第一部分（查并发）先做，结论回报后再决定是否修；第二部分照上面实现，每步先测后写。
