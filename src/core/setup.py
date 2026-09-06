"""首次启动配置引导与模型服务厂商预设。"""

import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.styles import Style

from .model_capabilities import extract_context_window

PROVIDER_HINTS = "↑/↓ move, Space select, Enter confirm, Esc cancel"


@dataclass(frozen=True)
class Vendor:
    """一个可选的模型服务厂商。"""

    name: str
    base_url: str


# 预设厂商与手动配置入口，base_url 为空表示手动输入
VENDORS = (
    Vendor("DeepSeek", "https://api.deepseek.com/"),
    Vendor("OpenAI", "https://api.openai.com/v1"),
    Vendor("Moonshot", "https://api.moonshot.cn/v1"),
    Vendor("Alibaba Bailian", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    Vendor("Zhipu", "https://open.bigmodel.cn/api/paas/v4"),
    Vendor("SiliconFlow", "https://api.siliconflow.cn/v1"),
    Vendor("Manual", ""),
)


def infer_provider(base_url: str) -> str:
    """根据 base_url 推断厂商名（小写），无法识别时返回空字符串。"""

    normalized = base_url.lower()
    if "deepseek" in normalized:
        return "deepseek"
    if "dashscope" in normalized or "aliyun" in normalized:
        return "alibaba"
    if "bigmodel" in normalized or "zhipu" in normalized:
        return "zhipu"
    if "siliconflow" in normalized:
        return "siliconflow"
    if "openai" in normalized:
        return "openai"
    if "moonshot" in normalized:
        return "moonshot"
    return ""


def list_models(base_url: str, api_key: str, timeout: float = 15.0) -> list[str] | None:
    """调用服务商的 /models 接口获取可用模型列表，失败时返回 None。"""

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        response = client.models.list()
        return [model.id for model in response.data]
    except Exception:
        return None


def discover_context_window(
    base_url: str,
    api_key: str,
    model_name: str,
    timeout: float = 15.0,
) -> int | None:
    """在首次配置时读取服务端模型窗口，服务端不提供则返回 None。"""

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        return extract_context_window(client.models.retrieve(model_name))
    except Exception:
        return None


async def write_settings_atomically(path: Path, data: dict) -> None:
    """原子写入 settings.json，先写临时文件再重命名，避免写半截文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


async def run_setup_guide(target_path: Path) -> bool:
    """执行交互式首次配置引导，完成则写入配置文件并返回 True。"""

    base_url = await _pick_vendor()
    if base_url is None:
        return False
    api_key = await _prompt_api_key()
    if api_key is None:
        return False
    models = await asyncio.to_thread(list_models, base_url, api_key)
    model_name = await _pick_model(models)
    if model_name is None:
        return False
    context_window = await asyncio.to_thread(
        discover_context_window,
        base_url,
        api_key,
        model_name,
    )
    if context_window is None:
        context_window = await _prompt_context_window()
        if context_window is None:
            return False
    await write_settings_atomically(
        target_path,
        {
            "model": {
                "base_url": base_url,
                "api_key": api_key,
                "model_name": model_name,
                "context_window": context_window,
            }
        },
    )
    print(f"Config written to {target_path}")
    return True


async def _pick_vendor() -> str | None:
    """选择模型服务厂商，选择手动配置时继续询问服务地址。"""

    choice = await _pick_single(
        f"Choose a model provider ({PROVIDER_HINTS})",
        [vendor.name for vendor in VENDORS],
    )
    if choice is None:
        return None
    vendor = next(vendor for vendor in VENDORS if vendor.name == choice)
    if vendor.base_url:
        return vendor.base_url
    return await _prompt_base_url()


async def _prompt_base_url() -> str | None:
    """手动输入服务地址，校验为 http/https 后返回，取消时返回 None。"""

    while True:
        value = await _prompt_text("Base URL (e.g. https://api.example.com/v1): ")
        if value is None:
            return None
        parsed_url = urlparse(value.strip())
        if parsed_url.scheme in {"http", "https"} and parsed_url.netloc:
            return value.strip()
        print("Invalid base URL. Please enter a valid http(s) URL.")


async def _prompt_api_key() -> str | None:
    """输入 API key，取消时返回 None。"""

    return await _prompt_text("API key: ", is_password=True)


async def _pick_model(models: list[str] | None) -> str | None:
    """选择默认模型；模型列表拉取失败时改为手动输入。"""

    if models:
        return await _pick_single(
            f"Choose the default model ({PROVIDER_HINTS})",
            models,
        )
    print("Could not fetch the model list from this provider.")
    value = await _prompt_text("Model name: ")
    if value is None:
        return None
    return value.strip() or None


async def _prompt_context_window() -> int | None:
    """服务端未声明模型窗口时要求用户显式输入。"""

    while True:
        value = await _prompt_text("Context window in tokens: ")
        if value is None:
            return None
        try:
            context_window = int(value)
        except ValueError:
            context_window = 0
        if context_window > 0:
            return context_window
        print("Context window must be a positive integer.")


async def _prompt_text(prompt_text: str, *, is_password: bool = False) -> str | None:
    """读取一行文本输入，Esc / Ctrl+C 取消时返回 None。"""

    try:
        return await asyncio.to_thread(
            prompt,
            prompt_text,
            is_password=is_password,
            key_bindings=_cancel_bindings(),
        )
    except (KeyboardInterrupt, EOFError):
        return None


def _cancel_bindings() -> KeyBindings:
    """为文本输入框提供 Esc 取消快捷键。"""

    key_bindings = KeyBindings()

    @key_bindings.add("escape")
    def cancel(event) -> None:
        """取消当前输入"""

        raise KeyboardInterrupt

    return key_bindings


async def _pick_single(title: str, items: Sequence[str]) -> str | None:
    """展示单选列表并返回选中的项，Esc 取消返回 None。"""

    cursor_index = 0
    selected = items[0] if items else None

    def render() -> AnyFormattedText:
        """渲染提示语和单选列表，选中行用 › 前缀。"""

        fragments: list[tuple[str, str]] = [("", f"{title}\n\n")]
        for index, item in enumerate(items):
            prefix = "› " if index == cursor_index else "  "
            style = "class:selected" if index == cursor_index else ""
            fragments.append((style, f"{prefix}{item}\n"))
        return fragments

    key_bindings = KeyBindings()

    @key_bindings.add("up")
    def move_up(event) -> None:
        """向上移动光标"""

        nonlocal cursor_index
        cursor_index = max(0, cursor_index - 1)
        event.app.invalidate()

    @key_bindings.add("down")
    def move_down(event) -> None:
        """向下移动光标"""

        nonlocal cursor_index
        cursor_index = min(len(items) - 1, cursor_index + 1)
        event.app.invalidate()

    @key_bindings.add("space")
    def select(event) -> None:
        """用 Space 选中当前项"""

        nonlocal selected
        if items:
            selected = items[cursor_index]
            event.app.invalidate()

    @key_bindings.add("enter")
    def confirm(event) -> None:
        """确认当前选中项"""

        if items:
            event.app.exit(result=selected)

    @key_bindings.add("escape")
    def cancel(event) -> None:
        """取消本次选择"""

        event.app.exit(result=None)

    application = Application(
        layout=Layout(
            Window(
                content=FormattedTextControl(render),
                wrap_lines=False,
            )
        ),
        key_bindings=key_bindings,
        style=Style.from_dict({"selected": "fg:ansiblue"}),
        full_screen=True,
    )
    return await application.run_async()
