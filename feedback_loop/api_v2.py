from __future__ import annotations

from typing import Any
import argparse
import json

from feedback_loop.defaults import DEFAULT_MODELS
from feedback_loop.evaluation import EvalResult, build_feedback_message, load_global_inputs
from feedback_loop.reporting import print_feedback_accuracy_table
from feedback_loop.requesting import send_requests_in_parallel
from feedback_loop.v2_config import MODEL_RESPONSE_DIRS, PROMPT_PATHS
from feedback_loop.v2_records import build_attempt_record
from feedback_loop.v2_state import TaskState
from graders.framework_v2 import FrameworkV2Evaluator, load_framework_v2_tasks
from graders.qiskit_v2_specs import QISKIT_V2_JSONL, QiskitV2Evaluator, load_qiskit_v2_tasks
from utils.get_function_signature_from_prompt import get_function_signature_from_prompt
from utils.parse_prompt_with_feedback import parse_prompt
from utils.parse_response import parse_response
from utils.read_jsonl import read_jsonl


def build_task_states(jsonl_path: str, models: list[str]) -> list[TaskState]:
    states: list[TaskState] = []
    for model in models:
        for task in read_jsonl(jsonl_path):
            prompt = task.get("prompt_v2") or task.get("complete_prompt", "")
            states.append(
                TaskState(
                    task_id=str(task["task_id"]),
                    entry_point=str(task["entry_point"]),
                    category=str(task.get("category")),
                    model=model,
                    prompt=prompt,
                    signature_prefill=get_function_signature_from_prompt(prompt) or "",
                )
            )
    return states


def build_request(state: TaskState) -> dict[str, Any]:
    history: list[dict[str, str]] = []
    for turn in state.history:
        history.append({"role": "assistant", "content": turn["assistant_code"]})
        history.append({"role": "user", "content": turn["feedback_to_model"]})
    return parse_prompt(
        prompt=state.prompt,
        chat_completion=state.signature_prefill,
        model=state.model,
        n=1,
        prefill=False,
        extra_messages=history or None,
    )


def build_evaluator(framework: str, inputs: dict[str, Any]):
    if framework == "qiskit":
        return QiskitV2Evaluator(load_qiskit_v2_tasks(QISKIT_V2_JSONL), inputs)
    return FrameworkV2Evaluator(framework, load_framework_v2_tasks(framework), inputs)


def extract_code(raw: dict[str, Any], state: TaskState) -> tuple[dict[str, Any] | None, str]:
    assistant_text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        parsed = parse_response((raw, state.signature_prefill), state.entry_point)
        code = parsed.get("code") or parsed.get("generated_code") or parsed.get("content") or assistant_text
        return parsed, code
    except Exception:
        return None, assistant_text


def evaluate_state(
    *,
    state: TaskState,
    code: str,
    evaluator: Any,
) -> EvalResult:
    execution, grader_details = evaluator.grade_code(
        task_id=state.task_id,
        code=code,
        entry_point=state.entry_point,
    )
    metric = grader_details["kl_value"] if "kl_value" in grader_details else grader_details.get("metric")
    passed = bool(grader_details["passed"])
    return EvalResult(
        compiled=True,
        ran=True,
        kl_div_result=metric,
        kl_div_bool=passed,
        output={"probabilities": execution.probabilities, "grader_details": grader_details},
        error=None if passed else json.dumps(grader_details)[:2000],
    )


def run_iteration(
    states: list[TaskState],
    *,
    framework: str,
    benchmark_version: str,
    evaluator: Any,
    feedback_num: int,
) -> list[dict[str, Any]]:
    pending = [state for state in states if not state.done and state.attempts_used < feedback_num]
    if not pending:
        return []
    requests = [build_request(state) for state in pending]
    records: list[dict[str, Any]] = []
    for state, request_payload, raw_response in zip(pending, requests, send_requests_in_parallel(requests)):
        parsed, code = extract_code(raw_response, state)
        state.attempts_used += 1
        state.last_code = code
        eval_res = evaluate_state(
            state=state,
            code=code,
            evaluator=evaluator,
        )
        feedback = "" if eval_res.kl_div_bool else build_feedback_message(eval_res)
        state.done = state.done or eval_res.kl_div_bool
        if feedback:
            state.last_feedback = feedback
            state.history.append({"assistant_code": code, "feedback_to_model": feedback})
        records.append(
            build_attempt_record(
                state,
                framework=framework,
                request_payload=request_payload,
                raw_response=raw_response,
                parsed_response=parsed,
                eval_res=eval_res,
                feedback=feedback,
                benchmark_version=benchmark_version,
            )
        )
    return records


def write_outputs(states: list[TaskState], records: dict[str, list[dict[str, Any]]], models: list[str], framework: str) -> None:
    output_dir = MODEL_RESPONSE_DIRS[framework]
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        model_name = model.replace("/", "_")
        (output_dir / f"{model_name}_{framework}_attempts.json").write_text(
            json.dumps(records[model], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        final = [state.__dict__ for state in states if state.model == model]
        (output_dir / f"{model_name}_{framework}_final.json").write_text(
            json.dumps(final, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def main(models: list[str], framework: str, feedback_num: int, benchmark_version: str) -> None:
    states = build_task_states(PROMPT_PATHS[(framework, benchmark_version)], models)
    inputs = load_global_inputs(framework)
    evaluator = build_evaluator(framework, inputs)
    records = {model: [] for model in models}
    for _ in range(feedback_num):
        batch_records = run_iteration(
            states,
            framework=framework,
            benchmark_version=benchmark_version,
            evaluator=evaluator,
            feedback_num=feedback_num,
        )
        for record in batch_records:
            records[record["model"]].append(record)
        if not batch_records:
            break
    write_outputs(states, records, models, framework)
    print_feedback_accuracy_table(states, records, feedback_num, models)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feedback loop LLM evaluation on quantum tasks")
    parser.add_argument("models", nargs="*", help="Model names to evaluate")
    parser.add_argument("--framework", default="cirq", choices=["cirq", "pennylane", "qiskit"])
    parser.add_argument("--feedback_num", type=int, default=5)
    parser.add_argument("--benchmark-version", choices=["v2"], default="v2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.models or DEFAULT_MODELS, args.framework, args.feedback_num, args.benchmark_version)
