# MCP Provider 后续增强

## 目标

让用户可以通过配置文件配置一个 stdio MCP Server，启动 epsilon 时自动发现并注册 MCP 工具

## 已完成

- `.env` 支持 `MCP_STDIO_COMMAND`、JSON 数组 `MCP_STDIO_ARGS` 和 `MCP_STDIO_PROVIDER_ID`
- 启动时以当前工作区为工作目录创建 `StdioMcpProvider`
- 应用层通过 `ToolManager.register_mcp_provider` 发现并注册工具
- TUI 退出或注册失败时关闭 MCP 子进程
- 配置、启动传递、工具发现和实际统一调用均有测试

## 待完成事项

- 在 TUI 中区分本地工具和 MCP 工具的摘要

## 暂不处理

- MCP Server 网络传输
- 多进程连接池
- 动态热加载配置
- MCP 工具输出截断
