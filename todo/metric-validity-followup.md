# 指令：指标效度修正的落地（加注 + 产物分层 + 冻结覆盖率线）

> 背景：`design/evaluation.md` 第十四节（指标效度）由设计侧直接写好；`problems/metric-validity-correction.md`
> 也已落盘。执行方负责把这两份的结论**落到文档与评测产物里**，并**冻结那条不可靠的线**。
> 本任务**不跑任何模型**（0 token 成本）。

## 一、给受影响的文档加注（按统一样式）

在下列位置**逐处加注**（样式见 `problems/metric-validity-correction.md` 第四节）：

| 文档 | 要加注的位置 |
|---|---|
| `problems/big-repo-read-compare.md` | "三档结果"表下方 + "扇出读没有覆盖率优势"结论附近 |
| `problems/oncall-read-compare.md` | "源码级覆盖率"表下方 + 相关结论处 |
| `problems/completion-gate.md` | 覆盖率 / `inconclusive` 相关结论处 |
| `problems/completion-gate-v3.md` | "4.91M / +1/9 / 判定负面"那段下方 |

**统一加注文本**：

```
> ⚠️ 指标效度提示：本节引用的"源码级覆盖率"基于**任意挑出的源码事实清单**，
> 命中主要反映运气与报告篇幅，**不得单独支撑结论**（见 design/evaluation.md 第十四节）。
> 本节中仅依赖该指标的判断视为**未证实**。
```

**同时保留原始数字**——不删证据，只加效力说明。

## 二、评测产物按主/辅指标分层

改评测输出（`evaluation/big_repo_read_score.py`、`read_summary_score.py`、`medium_corpus_gate.py`
及今后新增的评分器），让 `result.json` 结构变成：

```jsonc
{
  "primary_metrics": {            // 客观、可复现
    "deliverable_completeness": ...,   // 结构化要求逐项 pass/fail
    "fabrication_free": ...,           // 引用的路径/标识符存在性核验
    "objective_pass": ...,             // tsc / 测试 / 命令退出码
    "cost": { "tokens": ..., "wall_clock_seconds": ..., "tool_errors": ..., "retries": ... }
  },
  "secondary_metrics": {          // 下界，不得单独支撑结论
    "fact_coverage": {
      "rate": ...,
      "item_types": { "constant": 30, "mechanism": 10 },   // ← 必须带题型构成
      "caveat": "事实无穷多，命中主要反映运气与篇幅"
    }
  }
}
```

**硬要求**：`secondary_metrics` 里**必须带 `item_types`（题型构成）**——否则读者无法判断这把尺子的可靠度。

## 三、冻结"覆盖率"这条线

1. **不再就"覆盖率能否提高"跑任何新实验**（`todo/completion-gate-v3.md` 的第五、六节保留为历史记录，
   但**不再执行其中"再跑一次干净重跑"的建议**——那把尺子已弃用）；
2. 在 `problems/completion-gate-v3.md` 顶部加一行状态：
   > **状态：本线的结论已按 `problems/metric-validity-correction.md` 降级；不再跑新实验。**
3. 以后若要用"深度"作主指标，**必须换任务形态**（自带 pass/fail：测试 / 类型检查 / 结构化交付物），
   而不是再改这把清单。

## 四、提交（**需用户同意**）

建议**两个提交**：

```
docs(eval): 增加指标效度规范，并修正受影响的结论效力

- design/evaluation.md 新增第十四节：指标分层（主=客观判据 / 辅=覆盖率下界）与四条硬规则
- 新增 problems/metric-validity-correction.md：逐条标注保留 / 降级与理由
- 给 big-repo-read-compare / oncall-read-compare / completion-gate(v3) 加统一效力提示
  （原始数字保留，不删证据）
- 冻结"覆盖率"这条实验线
```

```
refactor(eval): result.json 按主/辅指标分层并报告清单题型构成

- 主指标：交付物完整性 / 无编造 / 客观通过项 / 成本
- 辅助指标：事实覆盖率（必须带 constant/mechanism 题型构成与 caveat）
- 更新相关测试
```

**一切 git 命令（含 commit 与 push）先经用户同意。**

## 五、验收

1. 四处加注到位，且**原始数字未被删改**；
2. `result.json` 新结构在测试里被断言（含 `item_types` 必填）；
3. `problems/completion-gate-v3.md` 顶部有"不再跑新实验"的状态行；
4. 全量测试通过（当前 900 项，以实际值为准）。

## 六、边界

- **本任务不跑任何模型**；
- **不许改任何历史数字**——只加注；
- **不许把覆盖率重新包装成主指标**；
- 今后新增评分器，**一律按第二节的结构输出**。
