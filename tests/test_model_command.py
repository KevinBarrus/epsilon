"""测试 /model 命令的模型切换与新建配置流程。"""

import json
from pathlib import Path

import pytest

from core.commands.model import MANUAL_INPUT_OPTION, NEW_CONFIG_OPTION, model_command_slash
from core.commands.registry import CommandContext
from core.config import Settings
from core.context import ContextManager, DEFAULT_CONTEXT_BUDGET
from core.model import ClientHolder
from core.session import Session
from core.skills import SkillManager

BASE_URL = "https://example.com"
API_KEY = "test-key"


class FakeClient:
    """实现 ModelClient 协议的测试客户端。"""

    async def stream_chat(self, messages):
        yield ""

    async def stream_response(self, messages, tools=(), thinking_level=None):
        return
        yield  # 使函数成为不产生事件的异步生成器


class FakeAgentLoop:
    """记录热切换调用的测试 AgentLoop。"""

    def __init__(self) -> None:
        self.swapped_clients: list[object] = []

    def swap_client(self, client) -> None:
        """记录被替换的客户端。"""

        self.swapped_clients.append(client)


class FakeContextManager:
    """记录预算与模型名更新的测试 ContextManager。"""

    def __init__(self) -> None:
        self.budget_updates = 0
        self.model_names: list[str] = []

    def update_budget(self, budget) -> None:
        """记录预算更新次数。"""

        self.budget_updates += 1

    def set_model_name(self, model_name: str) -> None:
        """记录模型名更新。"""

        self.model_names.append(model_name)


class FakeScreen:
    """按队列返回选择结果的测试界面。"""

    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []
        self.status_message: str | None = None
        self.choice_results: list[str | None] = []
        self.text_results: list[str | None] = []

    def add_entry(self, role: str, content: str) -> int:
        """记录展示条目。"""

        self.entries.append((role, content))
        return len(self.entries) - 1

    def set_status_message(self, message: str) -> None:
        """记录状态消息。"""


    def set_working(self, message: str | None, show_elapsed: bool = True) -> None:
        return None
        self.status_message = message

    async def request_choice_picker(
        self,
        items: list[str],
        title: str,
        extra_options: list[str] | None = None,
    ) -> str | None:
        """返回下一个预设的选择结果。"""

        return self.choice_results.pop(0) if self.choice_results else None

    async def request_text_input(
        self,
        title: str,
        is_password: bool = False,
    ) -> str | None:
        """返回下一个预设的输入结果。"""

        return self.text_results.pop(0) if self.text_results else None


def _make_context(tmp_path: Path) -> tuple[CommandContext, FakeAgentLoop, FakeScreen]:
    """构造带假依赖的命令上下文。"""

    screen = FakeScreen()
    settings = Settings(
        BASE_URL,
        "deepseek-v4-pro",
        API_KEY,
        context_window=100_000,
    )
    holder = ClientHolder(settings, FakeClient())
    loop = FakeAgentLoop()
    manager = FakeContextManager()
    context = CommandContext(
        screen=screen,
        session=Session(tmp_path),
        skill_manager=SkillManager(tmp_path, global_skills_dir=tmp_path / "global"),
        context_manager=manager,
        client_holder=holder,
        agent_loop=loop,
        project_dir=tmp_path,
    )
    return context, loop, screen


@pytest.mark.asyncio
async def test_model_command_switches_to_selected_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试选择已有模型后统一热切换客户端与预算。"""

    context, loop, screen = _make_context(tmp_path)
    screen.choice_results = ["deepseek-v4-flash"]
    monkeypatch.setattr(
        "core.commands.model.list_models",
        lambda base_url, api_key: ["deepseek-v4-pro", "deepseek-v4-flash"],
    )

    await model_command_slash.handler(context)

    assert len(loop.swapped_clients) == 1
    assert context.client_holder.settings.model_name == "deepseek-v4-flash"
    assert context.client_holder.settings.base_url == BASE_URL
    assert ("tool", "Switched to model: deepseek-v4-flash") in screen.entries
    assert context.context_manager.model_names == ["deepseek-v4-flash"]


@pytest.mark.asyncio
async def test_model_command_cancel_keeps_current_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试取消选择时不改变当前配置。"""

    context, loop, screen = _make_context(tmp_path)
    screen.choice_results = [None]
    monkeypatch.setattr(
        "core.commands.model.list_models",
        lambda base_url, api_key: ["deepseek-v4-pro"],
    )

    await model_command_slash.handler(context)

    assert loop.swapped_clients == []
    assert context.client_holder.settings.model_name == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_model_command_manual_input_when_list_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试模型列表拉取失败时降级为手动输入模型名。"""

    context, loop, screen = _make_context(tmp_path)
    screen.choice_results = [MANUAL_INPUT_OPTION]
    screen.text_results = ["manual-model"]
    monkeypatch.setattr(
        "core.commands.model.list_models",
        lambda base_url, api_key: None,
    )

    await model_command_slash.handler(context)

    assert len(loop.swapped_clients) == 1
    assert context.client_holder.settings.model_name == "manual-model"
    assert ("tool", "Switched to model: manual-model") in screen.entries


@pytest.mark.asyncio
async def test_model_command_new_config_writes_project_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试新建配置会写入项目级配置并热切换。"""

    context, loop, screen = _make_context(tmp_path)
    screen.choice_results = [NEW_CONFIG_OPTION, "DeepSeek", "deepseek-v4-pro"]
    screen.text_results = ["new-api-key", "100000"]

    def fake_list_models(base_url: str, api_key: str):
        if api_key == "new-api-key":
            return ["deepseek-v4-pro", "deepseek-v4-flash"]
        return ["deepseek-v4-pro"]

    monkeypatch.setattr("core.commands.model.list_models", fake_list_models)

    await model_command_slash.handler(context)

    project_settings = json.loads(
        (tmp_path / ".epsilon" / "settings.json").read_text(encoding="utf-8")
    )
    assert project_settings["model"] == {
        "base_url": "https://api.deepseek.com/",
        "api_key": "new-api-key",
        "model_name": "deepseek-v4-pro",
        "context_window": 100000,
    }
    assert len(loop.swapped_clients) == 1
    assert context.client_holder.settings.model_name == "deepseek-v4-pro"
    assert context.client_holder.settings.api_key == "new-api-key"
    assert context.client_holder.settings.base_url == "https://api.deepseek.com/"
    assert ("tool", "Switched to model: deepseek-v4-pro") in screen.entries


@pytest.mark.asyncio
async def test_model_command_new_config_manual_model_when_list_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试新建配置时拉取模型失败会手动输入模型名。"""

    context, loop, screen = _make_context(tmp_path)
    screen.choice_results = [NEW_CONFIG_OPTION, "Manual"]
    screen.text_results = [
        "https://custom.example.com/v1",
        "new-api-key",
        "custom-model",
        "100000",
    ]

    monkeypatch.setattr(
        "core.commands.model.list_models",
        lambda base_url, api_key: None,
    )

    await model_command_slash.handler(context)

    project_settings = json.loads(
        (tmp_path / ".epsilon" / "settings.json").read_text(encoding="utf-8")
    )
    assert project_settings["model"] == {
        "base_url": "https://custom.example.com/v1",
        "api_key": "new-api-key",
        "model_name": "custom-model",
        "context_window": 100000,
    }
    assert context.client_holder.settings.base_url == "https://custom.example.com/v1"
    assert context.client_holder.settings.model_name == "custom-model"
