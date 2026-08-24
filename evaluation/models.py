"""定义评测场景、断言和结果的数据结构"""

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

EvaluationType = Literal[
    "core-regression",
    "real-task",
    "online-special",
    "code-correctness",
]


@dataclass(frozen=True)
class EvaluationScenario:
    """描述一个可重复执行的评测场景"""

    name: str
    description: str


@dataclass(frozen=True)
class EvaluationAssertion:
    """记录一条评测断言及其失败原因"""

    name: str
    passed: bool
    message: str = ""


@dataclass(frozen=True)
class EvaluationResult:
    """保存单次场景运行的统计信息和断言结果"""

    scenario: str
    duration_ms: float
    evaluation_type: EvaluationType = "core-regression"
    run_id: str = field(default_factory=lambda: str(uuid4()))
    repetition: int = 1
    task_id: str | None = None
    source: str | None = None
    evaluation_group: str | None = None
    base_commit: str | None = None
    changed_files: tuple[str, ...] = ()
    model_requests: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    retries: int = 0
    compactions: int = 0
    estimated_tokens: int = 0
    actual_tokens: int | None = None
    persistence_degraded: bool = False
    error_category: str | None = None
    error_stage: str | None = None
    error_message: str | None = None
    model_request_durations_ms: tuple[float, ...] = ()
    events: tuple[dict[str, object], ...] = ()
    assertions: tuple[EvaluationAssertion, ...] = ()

    @property
    def passed(self) -> bool:
        """返回场景是否通过全部断言"""

        return all(assertion.passed for assertion in self.assertions)
