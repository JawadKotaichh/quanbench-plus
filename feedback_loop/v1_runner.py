from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from feedback_loop.defaults import CANONICAL_SOLUTIONS_DIR
from feedback_loop.evaluation import build_feedback_message, evaluate_generated_code, load_global_inputs
from feedback_loop.paths import get_jsonl_path, get_model_responses_dir, load_json_list
from feedback_loop.reporting import print_feedback_accuracy_table
from feedback_loop.requesting import build_requests_for_states, extract_assistant_text, send_requests_in_parallel
from feedback_loop.v1_state import TaskState, build_task_states
from utils.parse_response import parse_response


def main(models: list[str], framework: str, feedback_num: int = 5, prefill: bool = False) -> None:
    output_dir = get_model_responses_dir(framework)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_by_task = {
        str(solution["task_id"]): solution
        for solution in load_json_list(CANONICAL_SOLUTIONS_DIR)
    }
    states = build_task_states(get_jsonl_path(framework), models)
    inputs = load_global_inputs(framework)
    attempts = {model: [] for model in models}
    attempt_paths = _attempt_paths(output_dir, framework, models)

    for _ in range(feedback_num):
        batch = run_iteration(states, framework, canonical_by_task, inputs, feedback_num, prefill)
        for record in batch:
            attempts[record["model"]].append(record)
        if not batch:
            break

    _write_attempts(attempt_paths, attempts)
    _write_final_summaries(output_dir, framework, models, states)
    print_feedback_accuracy_table(states, attempts, feedback_num, models)


def run_iteration(
    states: list[TaskState],
    framework: str,
    canonical_by_task: dict[str, dict[str, Any]],
    inputs: dict[str, Any],
    feedback_num: int,
    prefill: bool,
) -> list[dict[str, Any]]:
    pending = [state for state in states if not state.done and state.attempts_used < feedback_num]
    if not pending:
        return []
    requests = build_requests_for_states(pending, prefill=prefill)
    records: list[dict[str, Any]] = []
    for state, request, raw in zip(pending, requests, send_requests_in_parallel(requests)):
        code, parsed, assistant_text = _extract_code(raw, state)
        state.attempts_used += 1
        eval_res = evaluate_generated_code(state.task_id, state.entry_point, code, framework, canonical_by_task, inputs)
        feedback = "" if eval_res.kl_div_bool else build_feedback_message(eval_res)
        state.done = state.done or eval_res.kl_div_bool
        if feedback:
            state.history.append({"attempt": state.attempts_used, "assistant_code": code, "feedback_to_model": feedback})
            state.last_feedback = feedback
        state.last_code = code
        records.append(_record_attempt(state, framework, request, raw, assistant_text, parsed, code, eval_res, feedback))
    return records


def _extract_code(raw: dict[str, Any], state: TaskState) -> tuple[str, dict[str, Any] | None, str]:
    assistant_text = extract_assistant_text(raw)
    try:
        parsed = parse_response((raw, state.signature_prefill), state.entry_point)
        code = parsed.get("code") or parsed.get("generated_code") or parsed.get("content") or assistant_text
        return code, parsed, assistant_text
    except Exception:
        return assistant_text, None, assistant_text


def _record_attempt(
    state: TaskState,
    framework: str,
    request: dict[str, Any],
    raw: dict[str, Any],
    assistant_text: str,
    parsed: dict[str, Any] | None,
    code: str,
    eval_res: Any,
    feedback: str,
) -> dict[str, Any]:
    return {
        "framework": framework,
        "model": state.model,
        "task_id": state.task_id,
        "entry_point": state.entry_point,
        "category": state.category,
        "attempt": state.attempts_used,
        "done_after_attempt": state.done,
        "request_payload": request,
        "raw_response": raw,
        "assistant_text": assistant_text,
        "parsed_response": parsed,
        "code": code,
        "evaluation": {
            "compiled": eval_res.compiled,
            "ran": eval_res.ran,
            "kl_div_bool": eval_res.kl_div_bool,
            "error": eval_res.error,
            "output": None if eval_res.output is None else str(eval_res.output)[:2000],
        },
        "feedback_sent_to_model": feedback or None,
    }


def _attempt_paths(output_dir: Path, framework: str, models: list[str]) -> dict[str, Path]:
    return {
        model: output_dir / f"{model.replace('/', '_')}_{framework}_attempts.json"
        for model in models
    }


def _write_attempts(paths: dict[str, Path], attempts: dict[str, list[dict[str, Any]]]) -> None:
    for model, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(attempts[model], indent=2, ensure_ascii=False), encoding="utf-8")


def _write_final_summaries(output_dir: Path, framework: str, models: list[str], states: list[TaskState]) -> None:
    for model in models:
        final = [_final_record(state, framework) for state in states if state.model == model]
        path = output_dir / f"{model.replace('/', '_')}_{framework}_final.json"
        path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")


def _final_record(state: TaskState, framework: str) -> dict[str, Any]:
    return {
        "framework": framework,
        "model": state.model,
        "task_id": state.task_id,
        "entry_point": state.entry_point,
        "category": state.category,
        "done": state.done,
        "attempts_used": state.attempts_used,
        "last_code": state.last_code,
        "last_feedback": state.last_feedback,
    }
