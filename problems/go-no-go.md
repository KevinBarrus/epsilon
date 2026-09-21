# 评测 go/no-go 判据

面向 DeepSeek 改造后的 B/B'/C 四档对比，先钉死判据，避免验收时产生争议。

## 一、运行开关

B、B'、C 通过评测入口的开关组合区分，同一套代码、同一任务集：

| 档位 | 防火墙 | 驱逐 |
|---|---|---|
| A（基线） | 改造前代码，单独检出运行 | — |
| B | 开 | 关 |
| B' | 开 | 开 |
| C | 关 | 关 |

命令行：

```bash
uv run python -m evaluation.swebench --confirm \
  --instance-id <instance-id> --max-tool-rounds 80            # B（默认防火墙开、驱逐关）
uv run python -m evaluation.swebench --confirm \
  --instance-id <instance-id> --max-tool-rounds 80 --eviction # B'
uv run python -m evaluation.swebench --confirm \
  --instance-id <instance-id> --max-tool-rounds 80 --no-firewall # C
```

`--thinking` 默认 high，需要固定其它档位时显式传入。

## 二、判据

- `actual_tokens` 降幅不低于 15%；含驱逐档（B'）预期 40% 以上
- 通过数不低于 A
- token 对比只在产出补丁的任务上做（无补丁的运行不参与 token 对比）
- 对比条件必须可复现：结果文件记录 `model_name`、`thinking`、`firewall_enabled`、`eviction_enabled`

## 三、环境失败处理

- `tool_rounds == 0` 的运行判定为环境失败
- 主流程对该任务原地重跑一次，参数与代码不变
- 重跑有工具回合则计入结果；仍为环境失败则归入 `environment` 分组，不计入通过率分母，报告单独成组展示

## 四、样本与措辞

- 12 题属小样本，结论只作为工程回归数据，不宣称全量 SWE-bench 成绩
- 选题波动 ±1–2 题视为噪声，通过数明显下降（≥2 题）才触发停手复核
