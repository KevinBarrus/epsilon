"""实现 /delete 命令：永久删除当前会话并退出。"""

from ..session_store import SessionStore
from .registry import CommandContext, SlashCommand


async def delete_command(context: CommandContext) -> None:
    """确认后删除当前会话文件（先 trash 后 unlink）并退出。"""

    if context.session.session_id is None:
        context.screen.add_entry("tool", "No session to delete")
        return
    choice = await context.screen.request_choice_picker(
        ["yes", "no"],
        "Delete this session permanently? (unrecoverable)",
    )
    if choice != "yes":
        return
    deleted = SessionStore(context.project_dir).delete_session(
        context.session.session_id
    )
    if deleted:
        context.screen.add_entry("tool", "Session deleted")
    else:
        context.screen.add_entry("tool", "Session file not found")
    context.session.mark_deleted()
    context.screen.application.exit()


delete_command_slash = SlashCommand(
    name="delete",
    description="Permanently delete the current session and exit",
    handler=delete_command,
)
