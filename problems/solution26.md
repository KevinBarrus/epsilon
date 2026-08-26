# 第二十六轮优化方案：让写后验证识别检查语义，而非任意成功命令

## 目标

- 防止 `pip list`、`pwd`、`ls` 等环境查询命令取消写后验证提醒
- 保持本地验证与官方正确性裁判分层
- 让评测状态表达“是否做过有意义的检查”，而非“是否运行过任意命令”
- 保持一次提醒、自然结束和工具轮次上限的原有边界

## 唯一方案

在 `WriteVerificationPolicy` 内部增加一个纯函数，基于 `run_command` 的 `command` 参数识别常见检查命令。函数先按 shell 连接符拆分命令，再按命令名和独立参数匹配，避免把 `echo test`、`ls test.txt` 一类参数文本误判为检查。只有这一类命令的结果才会影响写后验证状态；所有写后命令仍保留在轨迹中，便于排查。

```text
成功写文件
  ↓
需要有意义的验证
  ↓
run_command
  ├─ 信息查询（pwd、ls、pip list）→ 不改变验证状态
  └─ 检查命令（测试、静态检查、构建、语法检查）
       ├─ 成功 → 可以自然结束
       └─ 失败 → 最多追加一次固定提醒
```

不调用模型判断命令意图，也不新增可配置规则引擎。第一版使用集中维护的命令名与参数标记表：规则透明、可单元测试，新增一种常见检查命令只需在一处增加标记。

## 代码改动顺序

### 问题一：识别有检查语义的命令

1. 在 `src/core/end_policy.py` 中定义集中维护的检查命令标记，覆盖当前 Coding Agent 常见的测试、静态检查、构建和 Python 语法检查
2. 新增纯函数，从 `ToolCall.arguments["command"]` 读取字符串命令并判断是否包含检查语义
3. 明确将 `pwd`、`ls`、`find`、`git status`、`pip list`、`which` 等信息查询排除在验证证据外
4. 无法读取字符串参数时按“非验证命令”处理，宁可补充一次提醒，也不将未知命令误判为验证
5. 为测试命令、语法检查、环境查询、缺失参数分别增加单元测试

初始标记仅覆盖：`pytest`、`unittest`、`tox`、`nox`、`manage.py test`、`test` 子命令、`compileall`、`py_compile`、`ruff check`、`flake8`、`mypy`、`eslint`、`make test`、`cargo test`、`go test`。后续只在真实项目出现未识别的检查命令时扩充。

### 问题二：分开记录“写后命令”与“写后检查”

1. 保留 `post_write_command_results`，继续记录文件写入后的全部 `run_command` 结果，避免丢失排查轨迹
2. 为 `EndPolicySummary` 和 `AgentRunResult` 增加 `verification_command_results`，只记录被识别为检查命令的结果
3. `agent_verification_status` 改为依据 `verification_command_results` 计算：
   - 没有检查命令：`not-attempted`
   - 存在检查命令且最终检查失败：`failed`
   - 最后一条检查命令成功：`passed`
4. 不将该字段用于通过率，也不改变官方 Harness 的 `official_harness_status`
5. 保持 JSONL 兼容读取：旧结果缺失新字段时按空检查列表处理

### 问题三：调整一次提醒与回归测试

1. 写后只有信息查询命令时，模型自然结束前应得到一次“尚未运行相关检查”的提醒
2. 写后检查命令失败时，保留现有“阅读失败输出并重试或说明阻塞”的提醒
3. 写后检查命令成功时，不追加提醒
4. 工具轮次耗尽、取消和模型异常时不额外请求模型
5. 运行 `tests/test_agent_loop.py`、评测相关单元测试和完整测试集

## 验收标准

- `pip list`、`echo test`、`ls test.txt` 成功后自然结束仍会收到一次写后验证提醒
- `pytest` 或 `python -m py_compile` 成功后可自然结束
- 失败的检查命令最多引入一次提醒，不会形成循环
- 评测 JSONL 能同时保留全部写后命令和有检查语义的命令结果
- 11001 这类“本地检查失败、官方通过”仍被官方 Harness 判为通过
- 13321 重跑时，环境查询不再掩盖关键测试失败
- 完整测试集通过

## 完成标记

- [x] 问题一：识别有检查语义的命令
- [x] 问题二：分开记录写后命令与写后检查
- [x] 问题三：调整提醒与回归测试
