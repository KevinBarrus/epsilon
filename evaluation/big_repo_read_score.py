"""大语料只读实验（codex）的评分：按子系统的源码级清单 + 路径精确率。

清单按 codex 的**子系统**划分，合计 40 条 = 30 条"具名常量/字面值" + 10 条**机制类**问题。

机制类问题的答案必须跨文件把逻辑连起来（例如 fork 模式如何截断父线程历史、
工具并行如何判定、审批被拒后控制流怎么走），因此要求命中多个标识符（`required_hits >= 2`），
单个 grep 不足以作答。
**每一条的关键词都逐项验证过不出现在全仓 Markdown 文档里**
（验证方式：`grep -ril --include='*.md' --include='*.mdx' <关键词>` 命中 0 处），
因此覆盖率反映的是"读了多少代码"，而不是"读没读文档"。

主指标 = 命中的条目数 ÷ 30。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from .read_summary_score import (
    ChecklistItem,
    DEFAULT_PATH_PATTERN,
    normalize,
    precision as _precision,
)

# codex 是 Rust 仓库，路径渲染要认得 .rs / .toml / .sh 等后缀
CODEX_PATH_PATTERN = re.compile(
    r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:rs|toml|sh|c|h|py|ts|tsx|vue|md|json|jsonl|sql|ya?ml|txt|snap|lock)"
)

SUBSYSTEM_CHECKLIST: dict[str, tuple[ChecklistItem, ...]] = {
    "core": (
        ChecklistItem(
            "core_agents_probe",
            "读取项目文档时，祖先目录并行探测的上限是多少？",
            ("max_concurrent_ancestor_probes",),
        ),
        ChecklistItem(
            "core_agents_separator",
            "多份项目文档拼接时使用什么分隔标记？",
            ("--- project-doc ---",),
        ),
        ChecklistItem(
            "core_compact_user_tokens",
            "压缩请求里用户消息的 token 上限是多少？",
            ("compact_user_message_max_tokens",),
        ),
        ChecklistItem(
            "core_fork_mode",
            "子 Agent 的 fork 模式如何继承父线程历史？截断规则是什么？",
            ("spawnagentforkmode", "lastnturns", "truncate_rollout_to_last_n_fork_turns"),
            2,
        ),
        ChecklistItem(
            "core_tool_parallel",
            "工具调用的并行执行是如何判定与组织的？",
            ("tool_supports_parallel", "toolcallruntime"),
            2,
        ),
        ChecklistItem(
            "core_approval_denied",
            "审批被拒绝之后，工具执行的控制流如何变化？",
            ("reviewdecision::denied", "toolerror::rejected"),
            2,
        ),
        ChecklistItem(
            "core_auto_compact",
            "自动压缩在什么时机触发，窗口如何推进？",
            ("run_inline_auto_compact_task", "advance_auto_compact_window"),
            2,
        ),
    ),
    "tui": (
        ChecklistItem(
            "tui_event_capacity",
            "线程事件通道的容量是多少？",
            ("thread_event_channel_capacity",),
        ),
        ChecklistItem(
            "tui_recent_denials",
            "自动审查的拒绝记忆上限是多少条？",
            ("max_recent_denials",),
        ),
        ChecklistItem(
            "tui_jsonrpc_not_found",
            "客户端把 JSON-RPC 方法不存在映射成哪个错误码常量？",
            ("jsonrpc_method_not_found",),
        ),
    ),
    "app-server": (
        ChecklistItem(
            "app_exec_timeout_exit",
            "命令执行超时用的退出码常量是哪个？",
            ("exec_timeout_exit_code",),
        ),
        ChecklistItem(
            "app_output_chunk",
            "输出分片大小提示的常量是什么？",
            ("output_chunk_size_hint",),
        ),
        ChecklistItem(
            "app_input_too_large",
            "输入过大返回的错误码字符串是什么？",
            ("input_too_large",),
        ),
        ChecklistItem(
            "app_request_processors",
            "请求处理器是如何分派与并发执行的？哪些阶段必须串行？",
            ("requestprocessor", "commandexecrequestprocessor"),
            2,
        ),
    ),
    "exec-server": (
        ChecklistItem(
            "exec_scan_depth",
            "能力发现的最大扫描深度是多少？",
            ("max_scan_depth",),
        ),
        ChecklistItem(
            "exec_dirs_per_root",
            "每个根目录扫描的目录数上限是多少？",
            ("max_directories_per_root",),
        ),
        ChecklistItem(
            "exec_arg0_helper",
            "arg0 执行 helper 的专用命令行参数是什么？",
            ("--codex-run-as-arg0-exec-helper",),
        ),
    ),
    "mcp": (
        ChecklistItem(
            "mcp_stdout_limit",
            "MCP 子进程 stdout 单行字节上限是多少？",
            ("max_mcp_stdout_line_bytes",),
        ),
        ChecklistItem(
            "mcp_stderr_limit",
            "MCP 子进程 stderr 单行字节上限是多少？",
            ("max_mcp_stderr_line_bytes",),
        ),
        ChecklistItem(
            "mcp_elicitation_method",
            "elicit 请求使用的 MCP 方法名是什么？",
            ("elicitation/create",),
        ),
        ChecklistItem(
            "mcp_tool_name_normalize",
            "MCP 工具名是如何归一化并加上前缀的？重名怎么处理？",
            ("normalize_tools_for_model_with_prefix", "legacy_mcp_tool_name_prefix"),
            2,
        ),
    ),
    "plugin": (
        ChecklistItem(
            "plugin_schema_uri",
            "agent plugin 的 MCP schema URI 是什么？",
            ("agent-plugins.org/schemas",),
        ),
        ChecklistItem(
            "plugin_root_variable",
            "插件根目录通过哪个环境变量传入？",
            ("plugin_root",),
        ),
        ChecklistItem(
            "plugin_env_dedupe",
            "插件环境变量与请求头的大小写不敏感去重规则是什么？",
            ("duplicate case-insensitive agent plugins", "client_owned_http_headers"),
            2,
        ),
    ),
    "sandbox": (
        ChecklistItem(
            "sandbox_bwrap_exit",
            "内置 bwrap 摘要校验失败时的退出码常量是哪个？",
            ("bundled_bwrap_digest_verification_failure_exit_code",),
        ),
        ChecklistItem(
            "sandbox_unreadable_glob",
            "不可读 glob 匹配数量的上限是多少？",
            ("max_unreadable_glob_matches",),
        ),
        ChecklistItem(
            "sandbox_read_roots",
            "Linux 平台默认只读根目录列表的常量叫什么？",
            ("linux_platform_default_read_roots",),
        ),
        ChecklistItem(
            "sandbox_mode_resolve",
            "沙箱模式是如何解析并选择实现的？",
            ("resolve_windows_sandbox_mode", "sandbox_setup_is_complete"),
            2,
        ),
    ),
    "network-proxy": (
        ChecklistItem(
            "proxy_attribution_magic",
            "归属信息帧的魔数是什么？",
            ("cdxpxy1",),
        ),
        ChecklistItem(
            "proxy_attribution_len",
            "归属令牌的最大长度常量是哪个？",
            ("max_attribution_token_len",),
        ),
        ChecklistItem(
            "proxy_attribution_flow",
            "归属信息是如何在连接上写入与读取的？",
            ("write_attribution_frame", "read_attribution_token"),
            2,
        ),
    ),
    "config": (
        ChecklistItem(
            "config_project_doc_max",
            "项目文档的字节上限常量是哪个？",
            ("default_project_doc_max_bytes",),
        ),
        ChecklistItem(
            "config_reserved_providers",
            "保留的 model provider id 列表常量叫什么？",
            ("reserved_model_provider_ids",),
        ),
    ),
    "protocol": (
        ChecklistItem(
            "protocol_guardian_risk",
            "守护流程的风险等级枚举叫什么？",
            ("guardianrisklevel",),
        ),
        ChecklistItem(
            "protocol_guardian_action",
            "守护判定的动作枚举叫什么？",
            ("guardianassessmentaction",),
        ),
        ChecklistItem(
            "protocol_network_rule",
            "网络策略规则动作的枚举叫什么？",
            ("networkpolicyruleaction",),
        ),
    ),
    "persistence": (
        ChecklistItem(
            "store_rollout_line",
            "rollout 文件的单行字节上限常量是哪个？",
            ("max_rollout_line_bytes",),
        ),
        ChecklistItem(
            "state_queue_items",
            "队列条目上限常量是哪个？",
            ("max_queue_items",),
        ),
        ChecklistItem(
            "state_pinned_section",
            "置顶分组的固定标识常量叫什么？",
            ("pinned_thread_section_id",),
        ),
        ChecklistItem(
            "store_writer_lock",
            "同一线程的并发写入是如何协调的？锁目录与锁文件叫什么？",
            ("writerlockcoordinator", "coordination.lock"),
            2,
        ),
    ),
}


def coverage_by_subsystem(text: str) -> dict[str, object]:
    """按子系统计算源码级覆盖率。"""

    normalized = normalize(text)
    subsystems: dict[str, dict[str, object]] = {}
    matched_total = 0
    item_total = 0
    hit_subsystems: list[str] = []
    for name, items in SUBSYSTEM_CHECKLIST.items():
        matched: list[str] = []
        missing: list[str] = []
        for item in items:
            hits = sum(1 for keyword in item.keywords if keyword in normalized)
            (matched if hits >= item.required_hits else missing).append(item.key)
        matched_total += len(matched)
        item_total += len(items)
        if matched:
            hit_subsystems.append(name)
        subsystems[name] = {
            "matched": matched,
            "missing": missing,
            "matched_count": len(matched),
            "total": len(items),
            "rate": round(len(matched) / len(items), 4),
        }
    return {
        "subsystems": subsystems,
        "matched_count": matched_total,
        "total": item_total,
        "rate": round(matched_total / item_total, 4) if item_total else None,
        "hit_subsystems": hit_subsystems,
        "subsystem_count": len(SUBSYSTEM_CHECKLIST),
        "subsystem_hit_count": len(hit_subsystems),
        "subsystem_hit_rate": round(len(hit_subsystems) / len(SUBSYSTEM_CHECKLIST), 4),
    }


# 全部 40 条（30 常量题 + 10 机制题）
ALL_ITEMS: tuple[ChecklistItem, ...] = tuple(
    item for group in SUBSYSTEM_CHECKLIST.values() for item in group
)


def score(text: str, workspace: Path) -> dict[str, object]:
    """返回报告的评分（子系统源码级覆盖率 + 路径精确率）。"""

    return {
        "coverage_source": coverage_by_subsystem(text),
        "precision": _precision(text, workspace, CODEX_PATH_PATTERN),
    }


def coverage_for_subsystems(text: str, subsystems: Sequence[str]) -> dict[str, object]:
    """只统计指定子系统的覆盖率（用于"中等语料"子集实验）。"""

    normalized = normalize(text)
    items = [(name, item) for name, group in SUBSYSTEM_CHECKLIST.items() if name in subsystems for item in group]
    matched: list[str] = []
    missing: list[str] = []
    for _, item in items:
        hits = sum(1 for keyword in item.keywords if keyword in normalized)
        (matched if hits >= item.required_hits else missing).append(item.key)
    return {
        "subsystems": list(subsystems),
        "matched": matched,
        "missing": missing,
        "matched_count": len(matched),
        "total": len(items),
        "rate": round(len(matched) / len(items), 4) if items else None,
    }
