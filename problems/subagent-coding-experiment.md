# Scout 真实编码任务对照

## 目标

观察 Scout 工具接入真实代码修复任务后，Agent 是否自然委派，以及开关两档的正确率、Token 和耗时。结果是单次描述性对照，不作为稳定性能结论。

## 任务与执行

- SWE-bench Lite：`django__django-11001`
- T1 修复问题并通过官方 Harness；T2 添加回归测试并运行测试/Harness；T3 重构并再次运行测试/Harness。
- 两档使用 `deepseek-v4-pro`、`thinking=high`，各用独立工作区，基线代码相同；关闭驱逐、防火墙开启。
- 父 Agent 与 Scout 均没有固定工具轮数上限。Scout 仅在 on 档可用。
- off 结果：`evaluation-results/long-task-scout-off/long_task.jsonl`
- on 结果：`evaluation-results/long-task-scout-on/long_task.jsonl`

## 结果

| 阶段 | off 通过 | off 父 Token | off 耗时 | on 通过 | on 父 Token | on 耗时 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| T1 | 是 | 1,194,129 | 387.4 秒 | 是 | 1,050,869 | 299.5 秒 |
| T2 | 是 | 555,792 | 78.0 秒 | 是 | 734,950 | 102.1 秒 |
| T3 | 是 | 1,612,913 | 294.3 秒 | 是 | 633,908 | 95.7 秒 |
| 总计 | 3/3 | 3,362,834 | 759.7 秒 | 3/3 | 2,419,727 | 497.3 秒 |

两个结果中 `spawn_agent` 调用数都是 **0**。on 档可见 Scout 工具并注入了使用说明，但模型没有发起任何 Scout 调用；Scout Token 为 0。会话和阶段记录中均没有 `spawn_agent` 工具调用事件。

表面上 on 档父 Token 少 28.0%，耗时少 34.5%，但两个运行均未使用 Scout，差异不能归因于多 Agent。这次没有测出 Scout 被使用时的收益。任务阶段全通过只说明两次父 Agent 独立完成了工作。

## 工具轮数上限

Scout 先前的 8 轮限制来自第一版设计稿中的固定上限要求，不是 AgentLoop 的技术要求。它在之前的冒烟中导致 3 个 Scout 提前结束，因此本次真实评测取消了 Scout 固定轮数限制；长任务父 Agent 也默认不限制工具轮数。模型单请求仍受网络超时保护，父任务取消会回收运行中的 Scout。

## 结论

这是一次真实编码任务和官方验证对照，但不是 Scout 执行效果对照：模型在这个任务上自然选择了不委派。若要测 Scout 的直接收益，下一次实验任务需要有明确、相互独立的调查工作，并要求 on 档完成至少一次 Scout 调查；比较时必须把“强制委派”带来的额外提示和调用成本算进去。
