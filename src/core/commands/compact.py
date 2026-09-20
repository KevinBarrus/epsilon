"""实现 /compact 命令：手动压缩当前上下文。"""

from ..context import ContextSummaryError
from .registry import CommandContext, SlashCommand


async def compact_command(context: CommandContext) -> None:
    """对会话消息强制做一次上下文压缩并持久化记录。"""

    context.screen.add_entry("tool", "Compacting context…")
    try:
        result = await context.context_manager.build_for_model_result(
            context.client_holder.client,
            context.session.get_messages(),
            context.session.get_compactions(),
            force_compaction=True,
            evictions=context.session.get_evictions(),
        )
    except ContextSummaryError:
        context.screen.add_entry("tool", "Compaction failed")
        return
    if result.eviction is not None:
        context.session.add_eviction(result.eviction)
    if result.compaction is None:
        context.screen.add_entry("tool", "No compaction produced")
        return
    context.session.add_compaction(result.compaction)
    context.screen.add_entry("tool", f"Context compacted ({result.compaction.tokens_before} tokens before)")


compact_command_slash = SlashCommand(
    name="compact",
    description="Manually compact the conversation context",
    handler=compact_command,
)
