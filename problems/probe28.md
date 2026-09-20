# 第二十八轮真机探针记录：DeepSeek 思考模式 + 工具循环的 reasoning_content 回传

## 一、目的

方案一（reasoning_content 回传）此前只有官方文档依据，没有实测依据。本记录补齐真机三探针，回答三个问题：

1. 思考开启 + tools + 不回传 reasoning 是否稳定 400
2. 思考开启 + tools + 完整回传 reasoning 是否成功
3. 压缩式截断历史（旧 assistant 连同思维链被丢掉）+ tools + 只回传残留，是 200 还是 400

## 二、环境

- 端点：`https://api.deepseek.com/`
- 模型：`deepseek-v4-pro`（对比组：`deepseek-reasoner`）
- 思考参数：`extra_body={"thinking":{"type":"enabled"}}` + `reasoning_effort="high"`（与生产默认一致）
- 请求路径：真实调用 DeepSeek API，直接使用 `core.openai_client.OpenAICompatibleClient` 与线上同一套序列化逻辑
- 探针脚本为一次性脚本，未入库

## 三、结论摘要

| 探针 | 构造 | 结果 |
|---|---|---|
| 1 | tools + assistant(有 tool_calls、无 reasoning)，tool_call id 为**服务端签发** | **200** |
| 1-对照 | 同上，但 tool_call id 为**伪造** | **400**，错误文本：`The reasoning_content in the thinking mode must be passed back to the API.` |
| 2 | tools + 完整回传 reasoning | **200** |
| 3 | 压缩截断：丢弃旧 assistant（含思维链）与其工具结果，保留 system 摘要 + 较新工具轮 | **200** |
| 3-驱逐 | 在压缩视图上把工具输出替换为 artifact 占位符 | **200** |
| 3-降级 | 压缩视图 + 连保留轮的 reasoning 也丢掉 | **200** |

**核心发现：400 规则真实存在，但对"服务端自己签发的 tool_call id"豁免。**

判定依据的二值实验（唯一变量是 tool_call id）：

| 实验 | tool_call id | 是否带 reasoning | 结果 |
|---|---|---|---|
| A | 服务端签发 | 否 | 200 |
| B | 伪造 | 是 | 200 |
| C | 服务端签发（工具结果内容被改写） | 否 | 200 |
| D | 伪造 | 否 | 400 |

即：满足「id 被服务端认识」或「reasoning_content 已提供」任一条件即通过；两者都不满足才 400。

## 四、对既有判断的修正

`problem28.md` 把 reasoning_content 回传定性为"硬阻断：思考模式 + 工具循环第二次请求就 400"。**真机未复现该阻断**：

- 生产链路中 tool_call id 全部由服务端生成，因此始终落在豁免一侧，不回传不会 400
- 旁证：改造前代码的基线评测（`AgentLoop` 默认 `thinking_level="high"`，12 题、80 轮工具预算）跑出 7/12，若第二轮即 400 则基线无法运行

但仍应保留回传实现，理由：

1. 官方文档明确要求，当前豁免属于服务端行为，不可依赖（服务端重启、id 状态过期、后端实例切换都可能让豁免消失）
2. 保留推理连续性，模型后续轮次可复用此前思考
3. 代码已实现并测试通过，回退没有收益

## 五、对压缩与驱逐路径的决定

探针 3 的三个变体全部 200，说明：

- 服务端不校验请求中已不存在的历史消息；压缩丢弃旧 assistant 与思维链是协议安全的
- 工具输出改写为占位符（驱逐）不影响协议有效性
- 驱逐按"越阈值一次性批量改写、之后前缀重新稳定"的设计可以照常实现，不会引入 400
- 补充（合并 9/19 执行方实测）：400 形态不止一种。形态②：`[system, assistant(纯文本摘要，无
  tool_calls、无 reasoning), assistant(真实 id + reasoning + tool_calls), tool]` 连续相邻
  assistant → **400**，工具轮同时满足"id 被认识"与"带 reasoning"仍被拒，触发者是前一条
  纯文本 assistant——无 tool_calls 的 assistant 消息没有 id 豁免通道，thinking 模式下一旦
  后随 reasoning 轮即触发校验。工程红线：摘要/合成消息不得使用 assistant 角色（现有压缩用
  system 角色，实测 200；user 摘要亦 200）

## 六、对评测消融的影响

规划方此前判断"回传 vs 不回传"无法做消融（违约档若 400 则跑不完）。真机结果显示该模型两侧都能跑完，因此：

- 可以在 3–4 题上做"回传 vs 不回传"小样本专项，量化推理连续性对通过率与 token 的真实影响
- 优先级不变：仍排在 B/B'/C 主表之后

## 七、9/19 执行方补充实验（本记录合并）

同端点同模型，A 段裸 SDK 复现 + B 段生产代码端到端：

| 实验 | 构造 | 结果 |
|---|---|---|
| A-1 | system, user, assistant(tool_calls，无 reasoning), tool | 200 |
| A-2 | 同上，assistant 带 reasoning 回传 | 200 |
| A-3a | system, **assistant(纯文本摘要)**, assistant(reasoning+tool_calls), tool | **400**（形态②） |
| A-3b | system, **user(摘要)**, assistant(reasoning+tool_calls), tool | 200 |
| A-3s | system, **system(摘要)**（现压缩形态）, assistant(reasoning+tool_calls), tool | 200 |
| A-3g | 旧轮 assistant(tool_calls 无 reasoning) + 新轮 assistant(带 reasoning)，中间隔 tool/user 消息 | 200 |
| B | 生产代码 `OpenAICompatibleClient + AgentLoop` 真实工具循环，thinking=high | 200，无 400，两轮 assistant reasoning 均回传存档 |

A-3g 说明形态②要求 assistant 消息**相邻**；正常对话与工具链（tool/user 消息分隔）不受影响。

## 八、遗留与待确认

- 服务端 id 豁免的持久性未测（跨天/跨实例是否仍豁免），当前结论仅覆盖短时间、同端点
- `deepseek-reasoner` 在伪造 id + 无 reasoning 时也返回 200，未强制该规则；本记录以项目实际使用的 `deepseek-v4-pro` 为准
- 回传 reasoning 会让每轮请求携带思考文本，是否与配置 C 的缓存收益冲突，需在 B/B'/C 评测中观察
