from __future__ import annotations

import argparse

from feedback_loop.defaults import DEFAULT_MODELS
from feedback_loop.evaluation import (
    EvalResult,
    build_feedback_message,
    evaluate_generated_code,
    get_probs_cirq,
    get_probs_pennylane,
    get_probs_qiskit,
    load_global_inputs,
)
from feedback_loop.paths import get_jsonl_path, get_model_responses_dir, load_json_list
from feedback_loop.reporting import print_feedback_accuracy_table
from feedback_loop.requesting import (
    build_requests_for_states,
    extract_assistant_text,
    send_request,
    send_requests_in_parallel,
)
from feedback_loop.v1_runner import main
from feedback_loop.v1_state import TaskState, build_task_states

__all__ = [
    "EvalResult",
    "TaskState",
    "build_feedback_message",
    "build_requests_for_states",
    "build_task_states",
    "evaluate_generated_code",
    "extract_assistant_text",
    "get_jsonl_path",
    "get_model_responses_dir",
    "get_probs_cirq",
    "get_probs_pennylane",
    "get_probs_qiskit",
    "load_global_inputs",
    "load_json_list",
    "main",
    "print_feedback_accuracy_table",
    "send_request",
    "send_requests_in_parallel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feedback loop LLM evaluation on quantum tasks")
    parser.add_argument("models", nargs="*", help="Model names to evaluate")
    parser.add_argument("--framework", default="cirq", choices=["cirq", "pennylane", "qiskit"])
    parser.add_argument("--feedback_num", type=int, default=5, help="Max attempts per task")
    parser.add_argument("--benchmark-version", choices=["v1"], default="v1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.models or DEFAULT_MODELS, args.framework, args.feedback_num)
