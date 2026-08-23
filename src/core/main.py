"""epsilon 的程序启动入口"""

import asyncio
import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import ConfigError, default_user_config_path, load_settings
from .context import ContextBudget
from .balance import create_balance_provider
from .openai_client import OpenAICompatibleClient
from .session_picker import SessionPicker
from .session_store import SessionStore
from .session_store import SessionStoreError
from .setup import run_setup_guide
from .status import create_status_info
from .tools import StdioMcpProvider
from .ui import ChatExitInfo, run_chat


logger = logging.getLogger(__name__)


async def run(
    session_id: str | None = None,
    resume: bool = False,
    config_path: Path | None = None,
) -> ChatExitInfo | None:
    """加载配置、创建模型客户端并启动终端界面"""

    workspace = Path.cwd().resolve()
    if resume and session_id is None:
        summaries = SessionStore(workspace).list_sessions()
        session_id = await SessionPicker(summaries).pick()
        if session_id is None:
            print("No session selected, exiting")
            return

    if config_path is None and not default_user_config_path().is_file():
        completed = await run_setup_guide(default_user_config_path())
        if not completed:
            print("Setup incomplete, exiting")
            return
    settings = load_settings(user_config_path=config_path)
    client = OpenAICompatibleClient(settings)
    mcp_provider = (
        StdioMcpProvider(
            (settings.mcp_stdio.command, *settings.mcp_stdio.arguments),
            settings.mcp_stdio.provider_id,
            cwd=workspace,
        )
        if settings.mcp_stdio is not None
        else None
    )
    balance_provider = create_balance_provider(settings.base_url, settings.api_key)
    balance = await balance_provider.get_balance()
    status = create_status_info(settings.model_name, balance)
    return await run_chat(
        client,
        status,
        settings,
        workspace,
        session_id,
        context_budget=ContextBudget(
            settings.context_window,
            settings.reserve_tokens,
            settings.keep_recent_tokens,
        ),
        balance_provider=balance_provider,
        mcp_provider=mcp_provider,
        max_tool_rounds=settings.max_tool_rounds,
    )


def format_exit_summary(exit_info: ChatExitInfo | None) -> list[str]:
    """格式化界面退出后输出到终端的会话摘要"""

    if exit_info is None:
        return []
    if exit_info.usage_totals is None:
        lines = ["Token usage: unavailable"]
    else:
        usage = exit_info.usage_totals
        lines = [
            "Token usage (this run): "
            f"total={usage.total_tokens} input={usage.prompt_tokens} "
            f"cached={usage.cached_tokens} output={usage.completion_tokens}"
        ]
    if exit_info.session_id is not None:
        lines.append(
            f"To continue this session, run epsilon resume {exit_info.session_id}"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """处理启动阶段的错误并返回进程退出码"""

    parser = argparse.ArgumentParser(description="Start epsilon")
    parser.add_argument(
        "--config",
        type=Path,
        help="Use a specific config file and skip first-run setup",
    )
    subparsers = parser.add_subparsers(dest="command")
    resume_parser = subparsers.add_parser("resume", help="Resume a previous session")
    resume_parser.add_argument("session_id", nargs="?", help="Session ID to resume")
    args = parser.parse_args(argv)

    try:
        exit_info = asyncio.run(
            run(
                getattr(args, "session_id", None),
                resume=args.command == "resume",
                config_path=args.config,
            )
        )
        for line in format_exit_summary(exit_info):
            print(line)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except SessionStoreError as exc:
        print(f"Session error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nExited.")
    except Exception:
        logger.debug("Unexpected error during startup", exc_info=True)
        print(
            "Runtime error: unexpected failure, retry or check debug logs",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
