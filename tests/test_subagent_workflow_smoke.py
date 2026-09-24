import shlex
import sys

import pytest

from core.config import Settings
from core.model import TextDelta, ToolCall, ToolCallEvent, UsageEvent
from core.tools.command_executor import CommandExecution, HostCommandExecutor
from evaluation.subagent_workflow_smoke import (
    _prepare_workspace,
    run_subagent_workflow_smoke,
)


class WorkflowClient:
    """模拟父 Agent 按 Scout、Worker、Reviewer 顺序委派。"""

    def __init__(self) -> None:
        self.parent_calls = 0
        self.child_calls: dict[str, int] = {}
        self.parent_system_messages: list[str] = []

    async def stream_response(self, messages, tools=(), thinking_level=None):
        system_text = "\n".join(
            message.content for message in messages if message.role == "system"
        )
        role = next(
            (name for name in ("Scout", "Worker", "Reviewer") if f"你是 {name}" in system_text),
            None,
        )
        if role is None:
            self.parent_system_messages = [
                message.content for message in messages if message.role == "system"
            ]
            index = self.parent_calls
            self.parent_calls += 1
            if index < 3:
                name = ("spawn_agent", "spawn_worker", "spawn_reviewer")[index]
                yield ToolCallEvent(
                    ToolCall(
                        f"parent-{index}",
                        name,
                        {
                            "task": f"{name} 的小任务",
                            **(
                                {"context": "Worker 修改了 calculator.py；验证命令：python -m unittest -v"}
                                if name == "spawn_reviewer"
                                else {}
                            ),
                        },
                    )
                )
            else:
                yield TextDelta("Scout 定位、Worker 修改、Reviewer 验证均完成")
            yield UsageEvent(20, 5, 25)
            return

        index = self.child_calls.get(role, 0)
        self.child_calls[role] = index + 1
        if role == "Worker" and index == 0:
            yield ToolCallEvent(
                ToolCall(
                    "write-1",
                    "write_file",
                    {
                        "path": "calculator.py",
                        "content": "def multiply(a, b):\n    return a * b\n",
                    },
                )
            )
        elif role == "Reviewer" and index == 0:
            yield ToolCallEvent(
                ToolCall("read-1", "read_file", {"path": "calculator.py"})
            )
        elif role == "Reviewer" and index == 1:
            yield ToolCallEvent(
                ToolCall(
                    "command-1",
                    "run_command",
                    {"command": "python -m unittest -v"},
                )
            )
        else:
            summaries = {
                "Scout": "## Files Read\n- calculator.py\n## Key Evidence\n- calculator.py:2 addition bug\n## Conclusion\n- multiplication is implemented with addition",
                "Worker": "## Changes\n- calculator.py\n## Verification\n- not run\n## Result\n- fixed",
                "Reviewer": "## Passed\n- unit test passed\n## Findings\n- none\n## Evidence\n- python -m unittest -v: Ran 1 test",
            }
            yield TextDelta(summaries[role])
        yield UsageEvent(30, 10, 40)


class SuccessfulCommandExecutor:
    """提供确定的测试结果，不启动真实子进程。"""

    async def execute(self, command, cwd, timeout_seconds):
        return CommandExecution(b"Ran 1 test: OK", b"", 0)


@pytest.mark.asyncio
async def test_workflow_fixture_has_a_real_passing_unit_test(tmp_path) -> None:
    """测试真机冒烟所用的小任务能通过真实单元测试命令。"""

    _prepare_workspace(tmp_path)
    command = f"{shlex.quote(sys.executable)} -m unittest -v"

    executor = HostCommandExecutor()
    broken = await executor.execute(command, tmp_path, 30)
    assert broken.returncode != 0
    assert "FAILED" in broken.stderr.decode()

    (tmp_path / "calculator.py").write_text(
        "def multiply(a, b):\n    return a * b  # corrected\n", encoding="utf-8"
    )
    result = await executor.execute(command, tmp_path, 30)

    output = result.stdout.decode() + result.stderr.decode()
    assert result.returncode == 0
    assert "Ran 1 test" in output
    assert "OK" in output


@pytest.mark.asyncio
async def test_workflow_smoke_records_role_chain_tokens_and_approvals(tmp_path) -> None:
    """测试工作流评测记录委派顺序、用量和审批结果。"""

    (tmp_path / "calculator.py").write_text(
        "def multiply(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "test_calculator.py").write_text(
        "from calculator import multiply\n", encoding="utf-8"
    )
    settings = Settings(
        "https://example.com",
        "test-model",
        "key",
        context_window=10_000,
        reserve_tokens=1_000,
        keep_recent_tokens=2_000,
    )

    client = WorkflowClient()
    result = await run_subagent_workflow_smoke(
        tmp_path,
        client,
        settings,
        command_executor=SuccessfulCommandExecutor(),
    )

    assert result.task_completed is True
    assert result.delegation_order == (
        "spawn_agent",
        "spawn_worker",
        "spawn_reviewer",
    )
    assert result.delegation_counts == {
        "scout": 1,
        "worker": 1,
        "reviewer": 1,
    }
    assert result.total_actual_tokens == 340
    assert [approval["tool"] for approval in result.approvals] == [
        "write_file",
        "run_command",
    ]
    assert result.verification_results[0]["passed"] is True
    assert "calculator.py" in result.reviewer_context
    assert "python -m unittest -v" in result.reviewer_context
    assert result.reviewer_read_paths == ("calculator.py",)
    assert any(
        f"Current workspace root: `{tmp_path}`" in content
        for content in client.parent_system_messages
    )
