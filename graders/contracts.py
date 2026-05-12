from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class GradeContext:
    probabilities: Sequence[float]
    canonical_probabilities: Sequence[float] | None = None
    candidate_unitary: Any | None = None
    canonical_unitary: Any | None = None
    target_unitary: Any | None = None
    metadata: Mapping[str, Any] | None = None
    code: str | None = None


@dataclass
class ExecutionResult:
    probabilities: list[float]
    metadata: dict[str, Any]
    unitary: Any | None = None
    circuit: Any | None = None


class TaskExecutor(Protocol):
    def __call__(
        self,
        *,
        task_id: str,
        code: str,
        entry_point: str,
        inputs: dict[str, Any],
    ) -> ExecutionResult: ...


class V2Evaluator(Protocol):
    def grade_code(
        self,
        *,
        task_id: str,
        code: str,
        entry_point: str,
    ) -> tuple[ExecutionResult, dict[str, Any]]: ...
