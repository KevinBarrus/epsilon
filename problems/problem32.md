# 问题 32：模型接入渠道缺失——只有 API Key，没有订阅认证

## 一、背景与问题

Epsilon 的模型接入目前只有一条路：`config.py` 里的 `api_key` + `base_url`，所有模型都走 OpenAI-compatible API。首次启动引导也只让用户"选服务商 + 填 API key"。

这带来的真实问题：**拥有订阅的用户无法使用 Epsilon**。

- 智谱 GLM Coding Plan 用户：有订阅额度，但被迫去开 API key；
- OpenAI 会员（ChatGPT / Codex Plus）用户：有订阅额度，但 Epsilon 没有"登录认证"入口，只能调 API；
- Anthropic 订阅用户：同理。

而有订阅的用户是**大多数**，愿意"开 API key、按 token 付费"的是少数。模型接入渠道缺失，直接决定了用户愿不愿意用 Epsilon。

## 二、参考项目的做法

- **pi**：支持 GPT 订阅认证（OAuth 登录，而非 API key）；
- **oh-my-pi**：有完整认证体系——`auth-broker-config`、`auth-storage`、`credential-pin`、`oauth_callback`，支持订阅登录 + token 刷新 + 凭据安全存储。

它们都不是"所有模型必须走 API"，而是"API key + 订阅认证"双渠道。

## 三、要实现的目标

模型接入从单渠道扩展为双渠道：

1. **保留 API key 渠道**（OpenAI-compatible，一个 client 通吃兼容服务商）；
2. **新增订阅认证渠道**（OAuth 登录）：用户用自己的订阅账号认证，Epsilon 通过订阅渠道调用模型，不需要 API key。

## 四、现实边界与工程量（不要低估）

订阅认证不是"加个登录按钮"，而是独立的接入层模块：

1. 每个服务商的 OAuth 流程不同（OpenAI / 智谱 / Anthropic 各自一套）；
2. 订阅额度背后的**非标准 API**——ChatGPT/Codex 的订阅 API 不是 OpenAI-compatible 的 `/v1/chat/completions`，需要专门的 client 适配；
3. token 刷新、失效处理、凭据安全存储；
4. 不同服务商的订阅额度查询与展示。

## 五、落地顺序（独立立项，不打断当前工作）

1. 完成多 Agent 评测（查并发 + Writer/Reviewer + 压力测试）；
2. 新 benchmark 实验；
3. **本问题**：订阅认证的设计与实现；
4. 打包上传、本地安装、大任务压力测试。

本问题届时单独出 solution（设计 OAuth 渠道 + 适配至少一个订阅服务商）。
