"""长任务评测的阶段验证：容器内固定测试命令与官方 Harness 回归检查。"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from core.tools.command_executor import CommandExecutor

from .swebench import (
    EVALUATION_COMMAND_TIMEOUT_SECONDS,
    HarnessResult,
    SwebenchTask,
    create_patch,
    verify_patch,
)


# T2/T3 的固定测试命令，只在此处定义一次，避免散落到各阶段配置
ORDERING_TESTS_COMMAND = (
    "/opt/miniconda3/envs/testbed/bin/python tests/runtests.py ordering --verbosity 2"
)
VALIDATION_OUTPUT_TAIL_CHARS = 1_200


@dataclass(frozen=True)
class StageValidation:
    """保存单个长任务阶段的验证结论。"""

    kind: str
    passed: bool
    detail: str
    command: str | None = None
    exit_code: int | None = None
    resolved: bool | None = None
    environment_error: str | None = None


async def run_ordering_tests(
    executor: CommandExecutor,
    workspace: Path,
    *,
    timeout_seconds: float = EVALUATION_COMMAND_TIMEOUT_SECONDS,
) -> StageValidation:
    """在容器内执行固定 ordering 测试命令，退出码为 0 才算通过。"""

    execution = await executor.execute(
        ORDERING_TESTS_COMMAND,
        workspace,
        timeout_seconds,
    )
    passed = execution.returncode == 0
    return StageValidation(
        kind="ordering-tests",
        passed=passed,
        detail=(
            "exit code 0"
            if passed
            else _output_tail(execution.stdout, execution.stderr)
        ),
        command=ORDERING_TESTS_COMMAND,
        exit_code=execution.returncode,
    )


async def run_harness_check(
    task: SwebenchTask,
    baseline: Path,
    workspace: Path,
    result_root: Path,
    harness_python: str,
) -> StageValidation:
    """对当前工作区生成补丁并跑官方 Harness，作为硬回归检查。"""

    _, patch = await asyncio.to_thread(create_patch, baseline, workspace)
    verification = await asyncio.to_thread(
        verify_patch,
        task,
        patch,
        result_root / "harness",
        harness_python,
    )
    return _harness_validation(verification)


def _harness_validation(verification: HarnessResult) -> StageValidation:
    """把官方 Harness 结果转换为阶段验证结论。"""

    return StageValidation(
        kind="harness",
        passed=verification.passed,
        detail=(
            verification.environment_error
            or verification.diagnostic
            or ("resolved" if verification.passed else "官方 Harness 未通过")
        ),
        resolved=verification.passed,
        environment_error=verification.environment_error,
    )


def _output_tail(stdout: bytes, stderr: bytes) -> str:
    """截取失败命令输出的尾部，避免报告被长日志淹没。"""

    combined = (stdout or b"") + (stderr or b"")
    text = combined.decode("utf-8", errors="replace").strip()
    if len(text) <= VALIDATION_OUTPUT_TAIL_CHARS:
        return text
    return "…" + text[-VALIDATION_OUTPUT_TAIL_CHARS:]
