# Scout 多独立调查点冒烟评测

## 一、评测目的

旧任务让主 Agent 回答同一套运行时架构问题。即使打开 Scout，真机结果仍为 `scout_calls=0`，没有走到委派链路。

本轮把输入改为四个相互独立的调查点：

1. `config.py` 的配置加载流程；
2. `tools/permissions.py` 的权限判断；
3. `context.py` 的 token 估算；
4. `session_store.py` 的 JSONL 读写。

提示词明确说明四项可并行委派。目标是观察主模型是否会调用多个 Scout，并记录任务完成情况、父 Agent token、全部 Agent token、耗时、调用批次和 Scout 结果。

## 二、运行配置

- 模型：`deepseek-v4-pro`
- 思考强度：`high`
- 工作区：当前 Epsilon 仓库
- 父 Agent 最大工具轮次：30
- Scout 最大工具轮次：8
- Scout 并发上限：3
- 两次运行日期：2026-09-23
- off 原始结果：`evaluation-results/subagent-smoke-independent.jsonl`
- on 复验结果：`evaluation-results/subagent-smoke-independent-on-rerun.jsonl`

原始结果按仓库规则忽略、不提交。两档均为只读调查，没有修改工作区。off 档不向模型暴露 `spawn_agent`；on 档暴露该工具并注入 Scout 使用说明。

off 与 on 来自两次独立真机运行，因此这里只做单样本描述性对比，不把差异视为稳定性能。

## 三、usage 缺失根因

第一次 on 运行的 `scout_actual_tokens` 为 `null`。代码检查确认：

1. 项目配置的 `stream_usage=True`，Scout 请求会发送 `stream_options.include_usage`；父 Agent 同样通过该客户端收到 usage，因此不是 Scout 未开启流式 usage。
2. `TimedModelClient` 在请求开始时先记录 `None`，只有流末尾收到 `UsageEvent` 才替换。旧结果说明至少一个 Scout 请求没有走到这个末尾事件。
3. 旧 JSONL 没有保存请求是被取消、发生错误，还是正常结束但服务端没给 usage，因此无法继续确定第一次缺失属于 Scout 整体超时、请求错误重试或服务端遗漏。把其中任何一种写成既定根因都会超出证据。

评测器现已同时保存已收到的 usage 总和、缺失请求数和缺失原因。on 复验没有重现该问题：`scout_requests_missing_usage=0`，所有 Scout 模型请求都返回了 usage。因此第一次缺失的更具体原因仍不可恢复，但相同问题再次出现时能够直接定位。

## 四、复验结果

| 指标 | off | on 复验 | 对比 |
| --- | ---: | ---: | ---: |
| 任务完成 | 是 | 是 | 两档均完成 |
| Scout 调用次数 | 0 | **4** | 多 Scout 委派实际触发 |
| Scout 批次 | 0 | **1 个 parallel 批次，含 4 个调用** | **真并行** |
| Scout 结果 | — | 2 completed / 2 tool_limit | 一半达到 8 轮上限 |
| 父 Agent actual tokens | 339,813 | **189,743** | **减少 150,070（44.2%）** |
| Scout actual tokens | 0 | **326,894** | usage 全部收到 |
| 全部 Agent actual tokens | **339,813** | **516,637** | **增加 176,824（52.0%）** |
| usage 缺失请求 | 0 | **0** | 无缺失 |
| 总耗时 | 161.4 秒 | **839.9 秒（14.0 分钟）** | on 为 off 的 **5.2 倍** |
| 父模型请求数 | 11 | 11 | 相同 |
| 注入父上下文的 Scout 结果 | 0 字符 | 7,841 字符 | 4 个 Scout 结果合计 |

on 档的全部 Agent token 计算如下：

```text
189,743（父 Agent）+ 326,894（Scout）= 516,637 token
```

Scout 消耗占 on 档全部 token 的 63.3%。父 Agent 节省了 150,070 token，但 Scout 新增了 326,894 token，最终净增 176,824 token。

## 五、判断

### 5.1 是否真正并行

是。4 个 `spawn_agent` 位于同一个 `execution_mode=parallel` 工具批次，不是分散在四个父级批次中串行委派。

这里的“并行”指同一工具批次按并行模式调度。运行时并发上限为 3，所以 4 个 Scout 中最多三个同时运行，剩余一个会等待并发名额。

### 5.2 是省总 token，还是把 token 从父挪到子

本次结果是后者，而且总量被放大：父 Agent token 下降 44.2%，但全部 Agent token 增加 52.0%。Scout 确实隔离了父上下文的调查过程，却没有降低本次任务的整体 token 消耗。

### 5.3 15.6 倍耗时是否由串行委派造成

第一次 on 运行没有保存批次分组，因此不能追溯其 15.6 倍耗时是否包含串行委派。复验中 4 个调用位于同一个并行批次，但仍比 off 慢 5.2 倍；这说明串行委派不是出现明显延迟的必要条件，但不能排除它对第一次运行的影响。

第一次 on 运行耗时 42.1 分钟，复验为 14.0 分钟，单次时延存在较大波动。复验中还有 2 个 Scout 达到工具轮次上限；这是长耗时的相关观测，但现有数据没有分解每个 Scout 的模型时延，不能据此计算工具轮次上限对总耗时的具体贡献。

## 六、结论

- **委派有效性：通过。** 本次多独立调查点触发了 4 个 Scout。
- **并行性：通过。** 4 个 Scout 位于同一个 parallel 批次。
- **父上下文收益：本次为正。** 父 token 下降 44.2%。
- **全部 Agent token 收益：本次为负。** 总 token 增加 52.0%。
- **耗时收益：本次为负。** on 档耗时为 off 的 5.2 倍。

这次单样本结果支持“Scout 可以隔离父上下文”，但不支持“Scout 能节省整体 token 或缩短耗时”。第二版设计不能只看父 Agent token，应同时限制 Scout 调查轮次，并继续记录全部 Agent token、工具批次和各 Scout 结果。

## 七、context 传递与结构化返回复验

代码提交 `6a45877` 已加入可选 `context` 参数、Scout 结构化分节输出，以及评测中的 context 计数。随后按相同 on 档协议发起真机复验（父模型最大工具轮次 30，Scout 最大工具轮次 8）。

本次调用运行约 30 分钟仍未生成结果文件，最终中止时调用栈停在父 Agent 的 `search_files` 文件扫描中，退出码为 130。没有收到完整评测事件，因此以下指标均记为“无数据”，不与上一轮结果做数值对比，也不据此判断 context 优化有效或无效。

这次失败本身说明：仅在提示词中引导父模型传递最小 context，并不能保证父模型停止重复探索；在缺少评测器级硬超时的情况下，单个真机任务可能长时间占用调用。后续若继续复验，应先增加评测器级硬超时或降低父模型最大工具轮次，并单独记录中止原因。
