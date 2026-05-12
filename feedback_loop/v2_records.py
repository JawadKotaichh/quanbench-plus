from __future__ import annotations

from typing import Any

from feedback_loop.v2_state import TaskState


def build_attempt_record(
    state: TaskState,
    *,
    framework: str,
    request_payload: dict[str, Any],
    raw_response: dict[str, Any],
    parsed_response: dict[str, Any] | None,
    eval_res: Any,
    feedback: str,
    benchmark_version: str,
) -> dict[str, Any]:
    return {
        "framework": framework,
        "model": state.model,
        "task_id": state.task_id,
        "entry_point": state.entry_point,
        "category": state.category,
        "attempt": state.attempts_used,
        "done_after_attempt": state.done,
        "request_payload": request_payload,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "code": state.last_code,
        "evaluation": {
            "compiled": eval_res.compiled,
            "ran": eval_res.ran,
            "kl_div_bool": eval_res.kl_div_bool,
            "error": eval_res.error,
            "output": None if eval_res.output is None else str(eval_res.output)[:2000],
            "benchmark_version": benchmark_version,
        },
        "feedback_sent_to_model": feedback or None,
    }
