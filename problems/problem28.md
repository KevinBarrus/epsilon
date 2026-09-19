# 问题 28：DeepSeek 改造说明与代码事实存在出入，评测方案存在公信力缺口

## 一、问题背景

针对"Epsilon 改造为面向 DeepSeek 的 Coding Agent Harness"的交接说明，本轮做了三类核对：

1. 通读 `src/core` 相关模块（context、agent_loop、openai_client、cost、memory、tools/*、skills/*）与 `evaluation/`（swebench、online、models、fakes）
2. 对基线 `swebench-batch2-80-final` 与 `-cont` 的 `results.jsonl` 做全量统计（12 题、881 条工具结果逐条回放）
3. 查证 DeepSeek 官方文档（Thinking Mode、Context Caching 两页）

结论：说明的总体方向和优先级排序成立，评测红线（同条件对比、小样本措辞）是说明中最有价值的部分；但存在 3 处与事实的出入、1 个比说明预想更严重的硬阻断、1 个完全缺失的最大杠杆。

## 二、声明核对结果

| 说明中的声明 | 核对结果 |
|---|---|
| 7/12 通过、80 轮预算 | 属实：final 6/12 + cont 续跑 12708 通过 = 7/12。但 12 题中有 3 题（12708/14667/15902）首轮环境失败（tool_rounds=0），靠 -cont 目录续跑才补齐 |
| actual_tokens 100 万–370 万 | 基本属实：实测 78 万–371 万（13220 为 78 万，略低于下限） |
| 输出只有 16KB/400 行硬截断、原文丢弃 | 属实：`output_limits.py`，命令与文件工具各自调用 |
| memory.py 32 行、无长期记忆 | 属实 |
| skills 已按需加载 | 属实（默认只注入 name+description，激活后注入正文） |
| cached_tokens 已统计但评测未按任务记录 | 属实，且问题比说明写的更深（见问题二） |
| 压缩三级降级、工具链不切断 | 属实；且基线 12 题 compactions 全为 0——压缩从未触发，说明 84k 阈值内上下文一直塞得下，问题是"塞得下但越来越贵" |

## 三、问题一：思考模式 + 工具循环的 reasoning_content 回传是硬阻断

官方文档明确规定：**带 `tools` 参数的请求，之前所有轮次的 `reasoning_content` 必须完整回传，包括没有工具调用的轮次；不回传直接返回 400**。

当前实现只在流式时读取展示（`_reasoning_delta`），`Message` 没有该字段、`_serialize_message` 不发送。后果不是说明里写的"丢失推理连续性"这种软代价，而是：思考模式开启 + 工具循环时，**第二次请求就会 400**。

改动范围比说明预想大：`Message` 加字段 → JSONL 持久化与 restore → 序列化仅对 DeepSeek 发送（OpenAI 端会拒绝未知字段）→ `estimate_message_tokens` 计入 → 交互式 UI 的思考内容累计。其中"压缩丢弃旧 assistant 消息后是否仍触发 400"文档没有覆盖，必须真机验证。

连带缺口：思考模式默认开启（effort=high），`_apply_thinking_options` 在 `thinking_level=None` 时什么都不发送。评测矩阵没有固定 thinking 开关与模型名，reasoning 回传的效果无法归因，B/C/D 之间不可比。

## 四、问题二：缓存命中字段解析错误，配置 C 无数据支撑

官方 usage 返回 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`，而 `_build_usage_event` 只读 OpenAI 格式的 `prompt_tokens_details.cached_tokens`——对 DeepSeek 恒为 None。不修复它，配置 C 的缓存命中率字段永远是空，前缀缓存优化的效果无法证明。

另有两个连带问题：

1. `TimedModelClient.usages` 只存 int 总量，需要改存完整 `UsageEvent` 才能按任务汇总缓存命中
2. results.jsonl 目前不记录模型身份与开关状态，基线用的哪个模型都查不到

## 五、问题三：Firewall 8k 阈值只覆盖 2% 的输出，缺少最大的上下文杠杆

对基线 881 条工具结果的统计：p50=573 字符、p90=4022、超过 8k 的只有 18/881（2%），触及 16k 上限的仅 6 条。

对"工具输出在后续每一轮都重复计入输入"的逐条回放模拟：

| 策略 | 对工具输出贡献的累计输入的节省（12 题范围） |
|---|---|
| Firewall 8k 阈值 | 0%–42%，平均约 18%；两题（13590/14016）完全没有超 8k 的输出，节省为 0 |
| 只保留最近 10 条完整输出、更早的替换为占位符 | 36%–80% |

结论：Firewall 正确但不充分——它只截"大"的，而 token 消耗的主因是"旧的"。陈旧工具输出的批量驱逐与 Artifact Store 天然配对（旧输出替换为"见 artifact"，需要时用 `read_artifact` 取回），这才是 Pi"常驻内容做薄"的本意。

关键权衡：驱逐改写历史消息会打断 DeepSeek 前缀缓存。必须做成**越过阈值时一次性批量驱逐**（类似压缩边界），不能每轮渐进驱逐，否则配置 C 的命中率会被自己吃掉。相对地，Firewall 在工具结果进入历史时定型、之后不再改写，对前缀缓存是友好的。

## 六、问题四：四配置对比表的公信力缺口

1. 配置 A 定义有歧义："改造前代码（或全部开关关闭）"。reasoning 回传、序列化改动没有开关，若 A 用"新代码+开关全关"，A 已经不等于改造前。
2. `_configuration_record` 只记 `max_tool_rounds`，没有 model_name、thinking、firewall/pipeline 开关。
3. 基线 3/12 环境失败说明：±1–2 题的通过数波动可能纯粹是噪声，"掉了两个以上就停"的 go/no-go 判据会被误触发。
4. "actual_tokens 明显下降"没有数字，验收时必然产生争议。

有利条件：基线 12 题总时长 111 分钟（单题 3–16 分钟），周五晚跑 B、C 两遍约 4 小时，时间可行。

## 七、问题五：P1/P2 的范围与时间缺口

1. P1 在周六半天内做完"可评测"不现实；且三个子 Agent 各自独立上下文，D 配置的 actual_tokens 大概率不降反升，与表格叙事自相矛盾。
2. 80 轮预算如何在 Scout/Worker/Reviewer 之间记账没有定义，tool_rounds 不可比。
3. Artifact Store 是否跨子 Agent 共享没有说明：不共享则 Scout 的 evidence 传不到 Worker，Worker 只能重读，token 反而涨。
4. Reviewer 的 run_tests"只能执行任务声明的验证命令"——交互模式下"任务声明"从哪来没有定义，评测里又等于全放开。
5. 命令执行摘要取"前 20 行"价值存疑：Django/pytest 场景失败信息几乎都在尾部，前 20 行多数是环境噪声。
6. ContextReceipt 写入会话 JSONL 需要确认 restore 对未知 record 类型的兼容。
7. P2 的 revision 检查依赖 `git diff`，评测工作区是无 Git 历史的快照，需要降级路径。
8. P2 的 project_preference 召回对 SWE-bench 是死代码（任务彼此独立），为它写召回逻辑性价比最低。

## 八、本轮问题的本质

说明的杠杆选择只处理了"单条太大"，没有处理"总量变旧"；只计划测量缓存命中，但测量链路本身对 DeepSeek 是坏的；把一个 400 级协议阻断当成了一项普通优化。改造可以执行，但优先级和评测条件必须先修正，否则 go/no-go 检查点会建立在噪声和缺失数据上。
