# Completion Gate：实现、真机 smoke 与 codex 实验结论

**单次运行的描述性对照，不作统计推断。** 本文回答三件事：门有没有实现好、smoke 有没有过、
门在 codex 语料上有没有把覆盖率推上去。

## 一、头条结论（先说最要紧的）

**门在本次配置下没有起到作用。** 门后一共跑了 4 次（A 三次、B 一次），**每一次的拒绝原因都是
`inconclusive`**（验证器 token 预算耗尽），**没有一次是实质判定**；覆盖率没有提升
（A 门前 4/40 → 门后 2~4/40；B 门前 11/40 → 门后 5/40），而每次声明完成都要多付验证器的预算。

机制本身是好的（单测 19 项 + 真机 smoke 都过），**卡在"验证器在 83 万行语料上做不出判定"**。

## 二、门后 4 次运行（A×3、B×1） vs 门前基线

| A 档 | token | 源码级覆盖率 | 终止 | verified | 拒绝次数 | 其中 inconclusive |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| **门前**（原始基线） | 3,599,644 | **4/40** | completed | — | — | — |
| 门后 #1（验证器 2M） | 15,818,423 | 3/40 | **error**（模型返回非法 JSON 参数） | false | 2 | 2 |
| 门后 #2（验证器 3M） | 3,826,676 | 4/40 | completed（达上限放行） | **false** | 2 | 3 |
| 门后 #3（改成可核查标准） | 3,526,387 | 2/40 | completed（达上限放行） | **false** | 2 | 3 |


> ⚠️ 指标效度提示：本节引用的"源码级覆盖率"基于**任意挑出的源码事实清单**，
> 命中主要反映运气与报告篇幅，**不得单独支撑结论**（见 design/evaluation.md 第十四节）。
> 本节中仅依赖该指标的判断视为**未证实**。

- **覆盖率没有提升**：A 门前 4/40 → 门后 3/4/2（噪声范围内）；**B 门前 11/40 → 门后 5/40（明显更低）**。
  注意 B 门后只花了 17.1M（门前 38.0M），但它被门反复拒绝后仍以 `verified=false` 收尾，
  **没有换来更高覆盖率**；
- **门的判定全是 inconclusive**：四次运行的 `rejected_not_met = 0`、`rejected_inconclusive ≥ 2`，
  未满足项统一是"验证器 token 预算耗尽"——**门一次也没有做出实质判定**；
- **门的代价是确定的**：每次 `goal(op=complete)` 都要付验证器预算（2M/3M），而它没能拦住任何一次
  实质性的提前收手。

## 三、根因

验证器是一个 **fresh、只读的 Scout**（`run_readonly_audit`），它面对的是 **829,876 行**的 codex 语料：

- 它尝试"对照实际状态逐条验证"，去读代码 → 单个 Rust 文件可达几百 KB（最大 481 KB ≈ 十几万 token）→
  每次请求重发上下文 → **3M 预算在十几次请求内耗尽**；
- 预算耗尽被映射为 `inconclusive`，按设计"无法验证 ≠ 已完成"→ 拒绝 → 模型继续 → 再次声明 → 再耗尽
  → 直到拒绝上限（2 次）后放行并标 `verified=false`。

**这不是实现 bug，是"标准与预算没有按语料规模设计"**：在 83 万行语料上，"覆盖全部核心子系统"这类
标准天然需要一次全仓审计，而给验证器的预算是按"有界抽查"设的。

## 四、修法（三选一，都需要重新设计，不是调参）

1. **把验收标准改成"只对着交付物核查"**（本已试过一次，但仍不足）：标准要能靠"读报告 + 抽查少量文件"
   判定，例如"报告必须为每个子系统给出至少一段剖析，且每段标注的文件路径必须真实存在"——
   实测仍不够，因为验证器会顺手把报告里提到的每个路径都读一遍；
2. **显著抬高验证器预算并禁止读大文件**：例如预算 20M + 提示词禁止读取超过若干行/字节的文件，
   改用 `search_files` 定位后只读关键片段；
3. **换验证方式：不靠子 Agent 通读，靠脚本核查 + 模型只做判断题**：例如由 harness 算出
   "报告覆盖了哪些子系统"，再把统计结果交给一个小模型判定是否达标。

**在选定其中一条之前，不应再跑 B/C** —— 那只会把 3 倍的钱花在同一个无效判定上。

## 五、已完成的、确实站得住的部分

| 项 | 状态 | 证据强度 |
| --- | --- | --- |
| 完成门机制（三值判定 / 拒绝 / 上限放行 / 计数分类） | ✅ 实现 + 19 项单测通过 | 结构性（单测锁死） |
| **防绕过**：`observe_tool_results` 的同步兜底在配置验证器后失效 | ✅ 有专门测试锁死 | 结构性 |
| 真实模型闭环 smoke（拒绝 → 未满足项回注 → 补齐 → 通过） | ✅ 通过，30,702 token | 真机单次 |
| 门在 codex 语料上的有效性 | ❌ **未产生实质判定** | 三次运行一致（都是 inconclusive） |

Smoke 的关键数字：门在 attempt 1/2 判 `not_met` 并拒绝，模型在收到"未满足项"后把报告从
"只有 ALPHA-111"补齐到三个文件全覆盖，attempt 3 判 `met` 通过，`verified=true`。

## 六、顺带修掉的真实缺陷（都是这次真机暴露的）

1. **DeepSeek `reasoning_content` 400**：历史里的工具调用消息在服务端"id 豁免"失效时会 400。
   已改为**始终回传**（无推理时给空串，极小探针验证 200），并加测试。
   —— 这印证了 `problems/probe28.md` 的警告："豁免属于服务端行为，不可依赖"。
2. **harness 作用域 bug**：完成门构造块引用了尚未定义的回调（`UnboundLocalError`），已修。
3. **smoke 曾漏写权限**：`write_file` 全被审批拒绝导致模型空转 —— 已修（自动放行）。

## 七、异常与成本（如实记录）

- A #1 以 `error` 结束：模型返回了非法 JSON 工具参数（模型侧问题，非实现缺陷）；
- B 在门后完整跑完（17,136,073 token，已纳入上表）；**C 在发现验证器预算问题后被主动中止**，
  中止时已消耗 6,584,574 token，无 `result.json`，未纳入结论；
- 门后消耗：A#1 15.8M + A#2 3.8M + A#3 3.5M + B 17.1M + C(中止) 6.6M ≈ **46.8M token**，smoke 约 0.05M；
- **本轮没有产出可用的 C 档对照**，B 档虽跑完但同样全是 inconclusive，因此
  "门能否提高覆盖率"**没有答案**——而且现有数据显示门后 B 的覆盖率反而更低。

## 八、标注：门前后不可比

门开启后，评测行为发生了改变（`goal(op=complete)` 会被独立验证）。因此：

- `problems/big-repo-read-compare.md` 里的 A/B/C 结果全部是 **"完成门前"** 的数据；
- 此后的任何评测结果**与之前不可比**，必须显式标注是否开启完成门。

## 九、证据位置

- 完成门实现与测试：`src/core/goal.py`、`tests/test_completion_gate.py`（19 项）
- smoke：`evaluation/completion_gate_smoke.py`、`evaluation-results/completion-gate-smoke/`
- 门后三次 A 档与中止的 B：`evaluation-results/big-repo-read-gate/`
  （`attempt1a` / `attempt1b` / `attempt2` / `attempt3` / `attempt2b-provider`，后者含 B 与中止的 C）
- 门前基线：`evaluation-results/big-repo-read-probe-codex-2026-09-26/`

---

# 附篇：完成门 v2（证据包 + 脚本检查）与中等语料实测

> 上面是 v1 的记录。v1 的结论"门没用"**已被推翻一部分**——真因不是"验证器会探索"，
> 而是**验证器的预算用了共享总账**。下面是更正后的记录。

## 一、根因更正（v1 的 inconclusive 真因）

`evaluation/big_task_single_goal.py` 的 `BudgetedClient` 用**传入的账本**判断是否超预算：

```python
if self.limit is not None and self.ledger.total_tokens >= self.limit:
    raise TokenBudgetReached
```

而 v1 的验证器客户端是 `BudgetedClient(..., ledger, verifier_budget)`——用的是**主循环的共享总账**。
主循环早就超过 2M/3M/100k，于是**每次验证调用都在发请求之前就被判超预算**，
harness 把它映射成"验证器 token 预算耗尽"→ `inconclusive`。

**"验证器会通读仓库"只是风险，不是这次的实际原因**（v1 的 scout 连一次请求都没发出去）。

修法：给验证器**独立账本**做预算，同时外层再包一层 `UsageTrackingClient(..., ledger)`
把用量计入共享总账（评测要算钱）。

## 二、v2 实现（改动 ①②③）

| 改动 | 落地 |
| --- | --- |
| ① 有界证据包 | `src/core/completion_evidence.py`：整包 ≤14,000 字符（objective 4k / criteria 4k / 声称 2k / 改动 ≤100 条 4k / 命令 ≤100 条 4k / 脚本检查 4k / 尾巴 ≤5 条 4k），超限按优先级裁剪；改动与命令**来自动作记录** |
| ② 脚本化检查 | `path_exists` / `sections_cover` / `paths_per_section` / `files_changed` / `command_passed` / `mentioned_paths_exist`（通用，不泄露清单）；无脚本证据的条目明确标注，验证器对它们一律 `inconclusive` |
| ③ 独立终态 | 超限放行写 **`unverified`**，不再写 `complete` |
| 验证器模式 | 默认 **evaluator**：单次模型调用、**不给任何工具**；可选 subagent 模式带 `BoundedReadGuard`（≤3 文件 × ≤200 行） |

配置默认值随之回落到有界证据口径：`verifier_token_budget` 3M → **100k**、`timeout` 300s → **60s**。

## 三、smoke（验收 #2）

| 项 | 值 |
| --- | --- |
| 第一次声明 | `not_met`（脚本检查 FAIL）→ 拒绝，未满足项回注 |
| 第二次声明 | `met` → 通过，`verified=true` |
| 成本 | **9,778 token**（v1 同类 smoke 是 30.7k；v1 真机验证器是 2~3M） |

## 四、中等语料实测（改动 ④ 的关键一步）

语料：codex 的 `protocol` + `config` + `rmcp-client` + `codex-mcp` 四个 crate，
生产 Rust **143 文件 / 2.3 MB**；隐藏清单取 `protocol`+`config`+`mcp` 三组共 **9 条**。

| A 档 | token | 源码级覆盖率 | 报告 | 读调用 | verified | 拒绝 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| **门前** | 518,151 | **3/9 = 33.3%** | 9,435 字符 | 25 | false | 0 |
| **门后** | 1,138,011（2.2×） | **4/9 = 44.4%** | 14,316 字符 | 30 | **true** | 1 |

**门的那一次拒绝是实质的**：`mentioned_paths_exist` 判 FAIL——报告提到 68 个路径，
只有 23 个真实存在，编造了 `protocol.rs`、`protocol/src/mcp.rs` 等。模型随后修正，第二次声明通过。

**能得出的结论**：v2 的门**真的能判定、能拒绝、能促成修正**，而且成本从"几 M 且判不出来"
变成"约 4k/次且判得出"；覆盖率 **+11pp（3/9 → 4/9）**、报告变长 52%、代价 2.2× token。

> ⚠️ 指标效度提示：本节的"覆盖率 3/9 → 4/9"同样取自**任意挑出的源码事实清单**，
> 命中主要反映运气与报告篇幅，**不得单独支撑结论**（见 design/evaluation.md 第十四节）。
> 仅依赖该指标的判断视为**未证实**；本节的 token 成本与"能拦编造"不受影响。


**还不能得出的结论**：门**没有**把覆盖率推到"接近 100%"。原因是**门的 criteria 衡量的是结构合规
（三小节 + 每节 ≥2 个真实路径 + 无编造路径），而不是那 9 条隐藏清单**。
要回答"门能不能把覆盖率推到接近 100%"，需要把 criteria 设计成**能编码覆盖度的脚本检查**
（例如按子系统逐条断言关键事实）——这是下一步，不要拿现在的数据硬讲。

## 五、这一步顺带修掉的两个真问题

1. **`paths_per_section` 会撞前言里的同名单词**：模型在 `# protocol` 下写了 5 个真实路径，
   检查器却因为正文第一次出现 "protocol" 的地方在前言而判 0/2 → **误判 `not_met`**。
   已修（优先匹配标题行）并加测试；
2. **子集选错**：codex 没有 `codex-rs/mcp` crate（MCP 客户端在 `rmcp-client` / `codex-mcp`），
   已改用真实 crate 名做语料子集，隐藏清单仍按 `mcp` 分组统计。

## 六、证据位置

- v2 实现：`src/core/completion_evidence.py`、`src/core/goal.py`
- 测试：`tests/test_completion_evidence.py`（15 项）、`tests/test_completion_gate.py`（19 项）
- smoke：`evaluation/completion_gate_smoke.py`、`evaluation-results/completion-gate-smoke/`
- 中等语料：`evaluation/medium_corpus_gate.py`、`evaluation-results/medium-corpus-gate-2026-09-26/{nogate,gate}/`
