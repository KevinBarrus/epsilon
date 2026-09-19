# 第二十八轮优化方案：修正 DeepSeek 改造优先级，补齐陈旧输出驱逐与评测公信力

## 一、目标

- 把 reasoning_content 回传按硬阻断排期，并固定评测矩阵的模型与思考开关
- 修复 DeepSeek 缓存命中字段的解析，让配置 C 有数据支撑
- 增加陈旧工具输出的批量驱逐，作为 Firewall 之外的主杠杆
- 钉住配置 A 的代码版本，补齐评测结果的配置记录
- 量化 go/no-go 判据，增加环境失败的重跑规则
- 收敛 P1/P2 的范围：P1 降级为"可演示"，P2 砍掉 project_preference 召回实现

## 二、总体原则

保留原说明的 P0→P2 分层与评测红线，不做计划外扩张。本轮只做三类修正：

1. 优先级修正：协议阻断和测量链路修复先于功能
2. 杠杆补齐：在既有 Artifact Store 设计上叠加驱逐，不引入新基础设施
3. 评测条件钉死：所有配置跑在可复现的代码版本与开关组合上

## 三、改动方案

### 方案一（原 P0-2，升级为最高优先）：reasoning_content 回传

1. `Message` 增加 reasoning 字段；JSONL 持久化与 restore 同步支持
2. 序列化时仅对 DeepSeek 端点发送该字段，其他服务端不发送
3. `estimate_message_tokens` 将 reasoning 计入 token 估算
4. 真机验证两件事：不回传是否稳定 400；压缩截断历史后携带 tools 是否仍 400（文档未覆盖，以实测为准）
5. 评测矩阵显式固定 model_name 与 thinking 开关，写入 `_configuration_record`

### 方案二（原 P0-3，扩展）：缓存命中测量链路修复

1. `_build_usage_event` 增加 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 解析，保留 OpenAI 字段兼容
2. `TimedModelClient.usages` 改存完整 `UsageEvent`
3. `EvaluationResult` 增加 cached_tokens、缓存命中率、Receipt 数、pipeline 类型字段，`report.py` 同步汇总

### 方案三（新增 P0.5）：陈旧工具输出批量驱逐

1. 工具结果进入历史时先经 Firewall 定型，之后不再改写（保护前缀缓存）
2. 估算超过阈值（建议消息预算的一半）时，一次性把最近 N 条以外的工具输出替换为"见 artifact <id>"占位符，本次替换后前缀重新稳定
3. 不逐轮渐进驱逐；驱逐事件生成 ContextReceipt（source: eviction）
4. 评测矩阵增加一档：B' = Firewall + 驱逐，用于隔离两种杠杆的贡献

### 方案四：评测公信力

1. 配置 A 钉在改造前的 git 提交上运行，不用"新代码+开关全关"冒充基线
2. `_configuration_record` 记录 model_name、thinking、firewall/eviction/pipeline 开关
3. tool_rounds=0 的环境失败视为无效运行，当场重跑，不计入回归
4. go/no-go 量化为：actual_tokens 降幅 ≥15%（含驱逐时预期 40%+）且通过数不低于 A；token 对比只在产出补丁的任务上做
5. 时间安排可行：单配置一轮约 2 小时，周五晚 B、B'、C 串行可行

### 方案五：P1/P2 范围收敛

1. P1 定位从"可评测"降为"可演示"：D 配置只跑 1 遍，只报告通过数与人工观察；token 对比明确标注"三 Agent 独立上下文，总量不可与单 Agent 直接比较"
2. 80 轮预算在三个角色间显式记账（建议：Scout ≤15、Reviewer ≤15、剩余归 Worker）
3. 三个子 Agent 共享同一 Artifact Store（按项目目录，Receipt 按 session 归属）
4. Reviewer 的 run_tests 白名单来自评测/项目配置文件显式列出的命令，不做半套声明机制
5. 命令执行摘要砍掉"前 20 行"，预算让给尾部行数与 error 匹配行
6. ContextReceipt 使用独立 record type，Session restore 时跳过未知类型不报错
7. P2 revision 检查在无 Git 工作区降级为"标记 stale 原因未知并拒绝召回"
8. P2 只实现 working_context 与 validated_experience 的召回，project_preference 留 schema 占位

### 方案六（改造完成后启动）：事件流协议 + TS/Ink 独立前端

前提：DeepSeek 改造与评测表完成后再启动，不影响本轮改造排期；期间不动现有 TUI，旧界面保留为回退。

1. 以现有 `AgentEvent`（TextDelta/ToolCallEvent/UsageEvent/ToolExecutionEvent/ToolBatchEvent/RetryEvent）为基础，正式化为带版本号的 JSON 线协议：事件信封 `{v, seq, session_id, type, payload}`，NDJSON 逐行传输
2. 双向两个流：核心→前端的事件流（会话生命周期、模型增量、工具执行、用量、上下文事件、错误重试）；前端→核心的指令流（提交输入、审批决策、取消、slash 命令、会话列表与恢复）
3. 未知事件类型必须忽略，保证协议前后向兼容；版本号在进程握手时协商
4. 传输层首版用 stdio：前端 spawn 核心进程，与 MCP stdio 同模式，不引入端口和网络；后续可替换为 Unix socket 或 WebSocket
5. 线上事件与 Session JSONL 统一为同一格式：传输即持久化，天然获得录制回放与 golden 事件序列快照测试
6. TS/Ink 前端独立包维护：React 组件树渲染终端界面，事件→状态→视图单向数据流；`<Static>` 承接已完成内容回滚区（对应现有 inline_history），活动区对应现有 screen
7. 协议先行：先在 Python 侧补齐事件信封、序列化与回放测试，再实现前端

## 四、验收顺序

### 第一阶段：协议与测量（先于一切功能）

- DeepSeek 思考模式 + 工具循环不再 400；reasoning 回传仅对 DeepSeek 生效
- cached_tokens 与命中率按任务出现在 results.jsonl
- 连续两轮请求前缀序列化一致的测试通过

### 第二阶段：上下文杠杆

- 超阈值输出原文完整落盘；注入内容有界；read_artifact 可按范围取回
- 驱逐只在阈值边界触发一次；驱逐后前缀重新稳定
- B、B'、C 三档的 actual_tokens 对比达到量化预期

### 第三阶段：演示与边界

- Scout 工具集不含写工具；Handoff schema 拒绝超长和缺字段；Worker 锁拒绝并发
- 跨项目/revoked/过期 revision 不召回
- 全量单元测试通过；一题 SWE-bench 冒烟通过

### 第四阶段：事件流协议与 TS 前端（改造完成后启动）

- 协议信封与事件类型有 schema 定义和版本化握手
- 同一事件序列可在旧 TUI 与新前端分别回放，展示结果语义一致
- 新前端完成最小闭环：提交输入、审批工具、展示流式输出
- 旧 TUI 仍可独立运行

## 五、完成标记

- [ ] 方案一：reasoning_content 回传与评测矩阵固定
- [ ] 方案二：缓存命中测量链路修复
- [ ] 方案三：陈旧工具输出批量驱逐
- [ ] 方案四：评测公信力（配置 A 钉版本、心跳记录、go/no-go 量化）
- [ ] 方案五：P1/P2 范围收敛
- [ ] 方案六：事件流协议与 TS/Ink 前端（改造完成后启动）
