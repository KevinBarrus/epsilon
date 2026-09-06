"""实现 /model 命令：查看并切换模型，支持新建配置。"""

import asyncio
from dataclasses import replace

from ..config import Settings
from ..context import ContextBudget
from ..openai_client import OpenAICompatibleClient
from ..model_capabilities import resolve_model_capabilities
from ..setup import VENDORS, list_models, write_settings_atomically
from .registry import CommandContext, SlashCommand

NEW_CONFIG_OPTION = "new config"
MANUAL_INPUT_OPTION = "manual input"
_CHOICE_HINTS = "↑/↓ move, Enter confirm, Esc cancel"


async def model_command(context: CommandContext) -> None:
    """展示当前配置与可用模型，支持切换模型或新建配置。"""

    current = context.client_holder.settings
    context.screen.add_entry(
        "tool",
        f"Current config: {current.model_name} @ {current.base_url}",
    )
    models = await asyncio.to_thread(list_models, current.base_url, current.api_key)
    extra_options = [NEW_CONFIG_OPTION]
    if not models:
        models = []
        extra_options.append(MANUAL_INPUT_OPTION)
        context.screen.add_entry("tool", "Could not fetch model list, enter manually")
    choice = await context.screen.request_choice_picker(
        models,
        f"Select model ({_CHOICE_HINTS})",
        extra_options=extra_options,
    )
    if choice is None:
        return
    if choice == NEW_CONFIG_OPTION:
        await _create_new_config(context)
        return
    if choice == MANUAL_INPUT_OPTION:
        model_name = await context.screen.request_text_input("Enter model name")
        if model_name is None:
            return
        await _apply_model_switch(context, replace(current, model_name=model_name))
        context.screen.add_entry("tool", f"Switched to model: {model_name}")
        return
    await _apply_model_switch(context, replace(current, model_name=choice))
    context.screen.add_entry("tool", f"Switched to model: {choice}")


async def _create_new_config(context: CommandContext) -> None:
    """通过选厂商、输入 API key 新建项目级配置并热切换。"""

    current = context.client_holder.settings
    base_url = await _pick_vendor(context)
    if base_url is None:
        return
    api_key = await context.screen.request_text_input(
        "API key (Enter to confirm)",
        is_password=True,
    )
    if api_key is None:
        return
    models = await asyncio.to_thread(list_models, base_url, api_key)
    if models:
        model_name = await context.screen.request_choice_picker(
            models,
            f"Select default model ({_CHOICE_HINTS})",
        )
    else:
        context.screen.add_entry("tool", "Could not fetch model list, enter model name")
        model_name = await context.screen.request_text_input("Model name")
    if model_name is None:
        return
    settings = replace(
        current,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        context_window=None,
    )
    applied = await _apply_model_switch(context, settings)
    if applied is None:
        return
    model_config: dict[str, object] = {
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name,
    }
    if applied.context_window is not None:
        model_config["context_window"] = applied.context_window
    await _save_model_config(
        context.project_dir / ".epsilon" / "settings.json",
        model_config,
    )
    context.screen.add_entry("tool", f"Switched to model: {model_name}")


async def _save_model_config(path, model: dict) -> None:
    """把 model 配置合并写回 settings.json，保留 background 等其他字段。"""

    import json

    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    data["model"] = model
    await write_settings_atomically(path, data)


async def _pick_vendor(context: CommandContext) -> str | None:
    """选择模型服务厂商，手动配置时继续询问服务地址。"""

    choice = await context.screen.request_choice_picker(
        [vendor.name for vendor in VENDORS],
        f"Select provider ({_CHOICE_HINTS})",
    )
    if choice is None:
        return None
    vendor = next(vendor for vendor in VENDORS if vendor.name == choice)
    if vendor.base_url:
        return vendor.base_url
    return await context.screen.request_text_input(
        "Enter base URL (e.g. https://api.example.com/v1)"
    )


async def _apply_model_switch(
    context: CommandContext,
    settings: Settings,
) -> Settings | None:
    """统一替换客户端引用并更新上下文预算，不动已持久化的会话。"""

    new_client = OpenAICompatibleClient(settings)
    try:
        capabilities = await resolve_model_capabilities(settings, new_client)
    except Exception as exc:
        await new_client.close()
        value = await context.screen.request_text_input(
            "Provider did not report a context window; enter token limit"
        )
        if value is None:
            context.screen.add_entry("tool", f"Could not switch model: {exc}")
            return None
        try:
            context_window = int(value)
            if context_window <= settings.reserve_tokens:
                raise ValueError
        except ValueError:
            context.screen.add_entry("tool", "Context window must be a valid token count")
            return None
        settings = replace(settings, context_window=context_window)
        new_client = OpenAICompatibleClient(settings)
        capabilities = await resolve_model_capabilities(settings, new_client)
    context.client_holder.swap(settings, new_client)
    context.agent_loop.swap_client(new_client)
    context.context_manager.set_model_name(settings.model_name)
    context.context_manager.update_budget(
        ContextBudget(
            capabilities.context_window,
            settings.reserve_tokens,
            settings.keep_recent_tokens,
        )
    )
    return settings


model_command_slash = SlashCommand(
    name="model",
    description="View and switch models, or create a config",
    handler=model_command,
)
