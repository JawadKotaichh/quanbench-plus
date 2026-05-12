from __future__ import annotations
from feedback_loop.cirq_handlers import (
    counts_to_array_cirq,
    get_probs_dictionnary_cirq,
    task_6_input_cirq,
)
from feedback_loop.pennylane_handlers import (
    binary_array_to_decimal_pennylane,
    task_6_input_pennylane,
)
from feedback_loop.qiskit_handlers import (
    counts_to_array_qiskit,
    get_probs_dictionnary_qiskit,
    task_6_input_qiskit,
)
from utils.get_kl_div import get_kl_div
import argparse
import json
import os
import time
import traceback
from utils.parse_prompt_with_feedback import parse_prompt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import cirq
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from feedback_loop.defaults import NUMBER_OF_SHOTS
from utils.get_function_signature_from_prompt import get_function_signature_from_prompt
from utils.read_jsonl import read_jsonl
from utils.parse_response import parse_response
from utils.coda_local import is_coda_model, send_coda_request
from graders.qiskit_v2_specs import (
    QISKIT_V2_JSONL,
    QiskitV2Evaluator,
    load_qiskit_v2_tasks,
)

from feedback_loop.defaults import (
    DEFAULT_MODELS,
    GLOBAL_INPUTS,
    CANONICAL_SOLUTIONS_DIR,
)

from feedback_loop.framework_paths.paths_cirq import CIRQ_JSONL
from feedback_loop.framework_paths.paths_pennylane import PENNYLANE_JSONL
from feedback_loop.framework_paths.paths_qiskit import QISKIT_JSONL

from feedback_loop.framework_paths.paths_pennylane import (
    MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_PENNYLANE,
)
from feedback_loop.framework_paths.paths_cirq import (
    MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_CIRQ,
)
from feedback_loop.framework_paths.paths_qiskit import (
    MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_QISKIT,
)
from utils.common import get_handler

load_dotenv()


def get_jsonl_path(framework: str, benchmark_version: str = "v1") -> str:
    if framework == "cirq":
        return CIRQ_JSONL
    if framework == "pennylane":
        return PENNYLANE_JSONL
    if benchmark_version == "v2":
        return str(QISKIT_V2_JSONL)
    return QISKIT_JSONL


def get_model_responses_dir(framework: str) -> Path:
    if framework == "cirq":
        return MODEL_RESPONSES_DIR_CIRQ
    if framework == "pennylane":
        return MODEL_RESPONSES_DIR_PENNYLANE
    return MODEL_RESPONSES_DIR_QISKIT


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_global_inputs(framework: str) -> Dict[str, Any]:
    if framework == "cirq":
        GLOBAL_INPUTS["06"] = task_6_input_cirq()
    elif framework == "pennylane":
        GLOBAL_INPUTS["06"] = task_6_input_pennylane()
    else:
        GLOBAL_INPUTS["06"] = task_6_input_qiskit()
    return GLOBAL_INPUTS


def extract_assistant_text(raw: Dict[str, Any]) -> str:
    try:
        return raw["choices"][0]["message"]["content"]
    except Exception:
        return ""


def send_request(request_payload: Dict[str, Any]) -> Dict[str, Any]:
    if is_coda_model(request_payload.get("model")):
        return send_coda_request(request_payload)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY in environment.")
    print("Sending request to", request_payload.get("model"))
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    resp = requests.post(url, json=request_payload, headers=headers, timeout=180)
    return resp.json()


def send_requests_in_parallel(
    request_payloads: List[Dict[str, Any]],
    max_workers: int = 16,
) -> List[Dict[str, Any]]:
    results: List[Optional[Dict[str, Any]]] = [None] * len(request_payloads)
    total = len(request_payloads)
    done = 0

    print(f"   Sending {total} requests with {max_workers} concurrent workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_i = {
            ex.submit(send_request, payload): i
            for i, payload in enumerate(request_payloads)
        }

        for fut in as_completed(fut_to_i):
            i = fut_to_i[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                results[i] = {
                    "id": f"error-{int(time.time())}-{i}",
                    "choices": [
                        {
                            "finish_reason": "error",
                            "native_finish_reason": "error",
                            "message": {
                                "role": "assistant",
                                "content": f"Error: {exc}",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }

            done += 1
            if done % 10 == 0 or done == total:
                print(f"   Progress: {done}/{total} ({done / total * 100:.1f}%)")

            time.sleep(0.05)

    return [r if r is not None else {} for r in results]


@dataclass
class EvalResult:
    compiled: bool
    ran: bool
    kl_div_bool: bool
    kl_div_result: Optional[float] = None
    error: Optional[str] = None
    output: Optional[Any] = None


# -----------------------------------
# Get probabilities based on framwork
# -----------------------------------
## CIRQ ##
def get_probs_cirq(task_id, solution, entry_point, shots, inputs):
    circuit_or_counts = get_handler(task_id, solution, entry_point, inputs)
    if isinstance(circuit_or_counts, dict):
        counts = circuit_or_counts
    elif isinstance(circuit_or_counts, cirq.Circuit):
        counts = get_probs_dictionnary_cirq(circuit_or_counts, shots)
    else:
        raise TypeError(
            f"Expected CirqCircuit or dict, got {type(circuit_or_counts)} instead."
        )
    return counts_to_array_cirq(counts)


## PENNYALANE ##
def get_probs_pennylane(task_id, solution, entry_point, shots, inputs):
    circuit_or_counts = get_handler(task_id, solution, entry_point, inputs)
    if isinstance(circuit_or_counts, np.ndarray):
        batta = circuit_or_counts.tolist()
        if type(batta[0]) is list:
            counts = [0] * (2 ** len(batta[0]))
            for sample in batta:
                counts[binary_array_to_decimal_pennylane(sample)] += 1
            for j in range(len(counts)):
                counts[j] /= len(batta)
        elif type(batta[0]) is float:
            raise TypeError(
                "Model return expected value or sampled on a specified basis, wrong return type"
            )
        else:
            counts = [0, 0]
            for i in range(len(batta)):
                if batta[i] > 0:
                    counts[1] += 1
                else:
                    counts[0] += 1
            counts[0] /= len(batta)
            counts[1] /= len(batta)

    else:
        raise TypeError(f"Expected numpy array, got {type(circuit_or_counts)} instead.")
    return np.array(counts)


def get_probs_qiskit(task_id, solution, entry_point, shots, inputs):
    circuit_or_counts = get_handler(task_id, solution, entry_point, inputs)
    if isinstance(circuit_or_counts, dict):
        counts = circuit_or_counts
    elif hasattr(circuit_or_counts, "name"):
        counts = get_probs_dictionnary_qiskit(circuit_or_counts, shots)
    else:
        raise TypeError(
            f"Expected QuantumCircuit or dict, got {type(circuit_or_counts)} instead."
        )
    return counts_to_array_qiskit(counts)


def evaluate_generated_code(
    task_id: str,
    entry_point: str,
    code: str,
    framework: str,
    canonical_by_task,
    inputss=GLOBAL_INPUTS,
    benchmark_version: str = "v1",
    v2_evaluator: Optional[QiskitV2Evaluator] = None,
) -> "EvalResult":
    try:
        if benchmark_version == "v2":
            if framework != "qiskit" or v2_evaluator is None:
                return EvalResult(
                    compiled=True,
                    ran=False,
                    kl_div_bool=False,
                    error="benchmark_version=v2 is currently implemented for qiskit only",
                )
            execution, grader_details = v2_evaluator.grade_code(
                task_id=task_id,
                code=code,
                entry_point=entry_point,
            )
            metric = (
                grader_details["kl_value"]
                if "kl_value" in grader_details
                else grader_details.get("metric")
            )
            return EvalResult(
                compiled=True,
                ran=True,
                kl_div_result=metric,
                kl_div_bool=bool(grader_details["passed"]),
                output={"probabilities": execution.probabilities, "grader_details": grader_details},
                error=None if grader_details["passed"] else json.dumps(grader_details)[:2000],
            )
        match framework:
            case "cirq":
                output = get_probs_cirq(
                    task_id, code, entry_point, NUMBER_OF_SHOTS, inputss
                )
            case "pennylane":
                output = get_probs_pennylane(
                    task_id, code, entry_point, NUMBER_OF_SHOTS, inputss
                )
            case "qiskit":
                output = get_probs_qiskit(
                    task_id, code, entry_point, NUMBER_OF_SHOTS, inputss
                )
            case _:
                return EvalResult(
                    compiled=True,
                    ran=False,
                    kl_div_bool=False,
                    error=f"Unknown framework '{framework}'. Expected one of: cirq|pennylane|qiskit",
                )
    except SyntaxError:
        return EvalResult(
            compiled=False, ran=False, kl_div_bool=False, error=traceback.format_exc()
        )
    except Exception:
        return EvalResult(
            compiled=True, ran=False, kl_div_bool=False, error=traceback.format_exc()
        )

    canonical = canonical_by_task.get(task_id)
    canonical_probs = canonical.get("canonical_output")

    try:
        if len(output) != len(canonical_probs):
            return EvalResult(
                compiled=True,
                ran=True,
                kl_div_bool=False,
                output=output,
                error=f"shape mismatch: model_probs len {len(output)},canonical_probs len {len(canonical_probs)}",
            )
        kl_value, kl_div_bool = get_kl_div(
            probs=output,
            expected_probs=canonical_probs,
        )
        return EvalResult(
            compiled=True,
            ran=True,
            kl_div_result=kl_value,
            kl_div_bool=kl_div_bool,
            output=output,
            error=None,
        )
    except Exception:
        return EvalResult(
            compiled=True,
            ran=True,
            kl_div_bool=False,
            output=output,
            error="Failed while comparing output to expected:\n"
            + traceback.format_exc(),
        )


def build_feedback_message(eval_res: EvalResult) -> str:
    if not eval_res.compiled:
        return (
            "Your previous code did not compile.\n\n"
            "Here is the error:\n"
            f"{eval_res.error}\n\n"
            "Please fix the code and respond with the FULL corrected code."
        )
    if eval_res.compiled and not eval_res.ran:
        return (
            "Your code compiled but failed at runtime.\n\n"
            "Here is the error:\n"
            f"{eval_res.error}\n\n"
            "Please fix the issue and respond with the FULL corrected code."
        )
    return (
        "Your code ran, but the output does NOT match the expected canonical output.\n\n"
        f"Got (stringified): {str(eval_res.output)[:2000]}\n\n"
        "Please try again and respond with the FULL corrected code."
    )


# -----------------------------
# Task state + loop
# -----------------------------
@dataclass
class TaskState:
    task_id: str
    entry_point: str
    category: str
    model: str
    prompt: str
    signature_prefill: str
    attempts_used: int = 0
    done: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_code: str = ""
    last_feedback: str = ""


def build_task_states(jsonl_path: str, models: List[str], benchmark_version: str = "v1") -> List[TaskState]:
    tasks = read_jsonl(jsonl_path)
    states: List[TaskState] = []

    for model in models:
        for task in tasks:
            prompt = task.get("prompt_v2") if benchmark_version == "v2" else None
            prompt = prompt or task.get("complete_prompt", "")
            sig = get_function_signature_from_prompt(prompt) or ""
            states.append(
                TaskState(
                    task_id=str(task.get("task_id")),
                    entry_point=str(task.get("entry_point")),
                    category=str(task.get("category")),
                    model=model,
                    prompt=prompt,
                    signature_prefill=sig,
                )
            )
    return states


def build_requests_for_states(
    states: List[TaskState], prefill: bool = False
) -> List[Dict[str, Any]]:
    reqs: List[Dict[str, Any]] = []

    for st in states:
        extra_messages: List[Dict[str, str]] = []

        # Re-create conversation history as alternating assistant/user messages
        for turn in st.history:
            extra_messages.append(
                {"role": "assistant", "content": turn["assistant_code"]}
            )
            extra_messages.append(
                {"role": "user", "content": turn["feedback_to_model"]}
            )

        req = parse_prompt(
            prompt=st.prompt,
            chat_completion=st.signature_prefill,
            model=st.model,
            n=1,
            prefill=prefill,
            extra_messages=extra_messages if extra_messages else None,
        )
        reqs.append(req)

    return reqs


def print_feedback_accuracy_table(states, attempt_records, feedback_num, models):
    """
    Prints cumulative compilation + pass rate at each feedback level.

    - states: List[TaskState] from your pipeline (used to get total tasks per model)
    - attempt_records: Dict[str, List[record]] exactly like in your main()
    - feedback_num: max attempts (same value you pass to main)
    - models: list of model strings
    """
    print("\n" + "=" * 85)
    print("FEEDBACK RESULTS TABLE")
    print("=" * 85)

    for model in models:
        # Total tasks for this model (from states)
        model_states = [s for s in states if s.model == model]
        total_tasks = len(model_states)
        if total_tasks == 0:
            continue

        # Attempts list for this model
        model_attempts = attempt_records.get(model, [])
        # Build per-task first compiled/pass feedback level
        per_task = {
            s.task_id: {"first_compiled_level": -1, "first_accurate_level": -1}
            for s in model_states
        }

        for rec in model_attempts:
            task_id = str(rec.get("task_id", ""))
            attempt = int(rec.get("attempt", 0))
            ev = rec.get("evaluation", {}) or {}

            if task_id not in per_task or attempt <= 0:
                continue

            level = attempt - 1  # level 0 = first attempt (no feedback yet)

            compiled = bool(ev.get("compiled", False))
            passed = bool(ev.get("kl_div_bool", False))

            if compiled and per_task[task_id]["first_compiled_level"] == -1:
                per_task[task_id]["first_compiled_level"] = level

            if passed and per_task[task_id]["first_accurate_level"] == -1:
                per_task[task_id]["first_accurate_level"] = level

        print(f"\nModel: {model}")
        print("-" * 85)
        print(f"{'Feedback Level':<20} {'Compiled':<25} {'Accurate (Pass)':<25}")
        print("-" * 85)

        # cumulative at each feedback level
        for level in range(feedback_num):
            cumulative_compiled = sum(
                1
                for v in per_task.values()
                if v["first_compiled_level"] != -1
                and v["first_compiled_level"] <= level
            )
            cumulative_accurate = sum(
                1
                for v in per_task.values()
                if v["first_accurate_level"] != -1
                and v["first_accurate_level"] <= level
            )

            compile_rate = (cumulative_compiled / total_tasks) * 100
            accuracy_rate = (cumulative_accurate / total_tasks) * 100

            level_label = f"{level} feedback" if level > 0 else "0 (no feedback)"
            print(
                f"{level_label:<20} "
                f"{cumulative_compiled}/{total_tasks} ({compile_rate:.1f}%)"
                f"{'':<10} "
                f"{cumulative_accurate}/{total_tasks} ({accuracy_rate:.1f}%)"
            )

        print("-" * 85)

    print("=" * 85 + "\n")


def main(
    models: List[str],
    framework: str,
    feedback_num: int = 5,
    prefill: bool = False,
    benchmark_version: str = "v1",
):
    if benchmark_version == "v2" and framework != "qiskit":
        raise ValueError("benchmark_version=v2 is currently implemented for qiskit only.")
    model_response_dir = get_model_responses_dir(framework=framework)
    model_response_dir.mkdir(parents=True, exist_ok=True)
    canonical_solutions = load_json_list(path=CANONICAL_SOLUTIONS_DIR)
    canonical_by_task: Dict[str, Dict[str, Any]] = {
        str(sol["task_id"]): sol for sol in canonical_solutions
    }
    attempt_json_paths: Dict[str, Path] = {}
    attempt_records: Dict[str, List[Dict[str, Any]]] = {}

    for m in models:
        model_name = m.replace("/", "_")
        attempt_json_paths[m] = (
            model_response_dir / f"{model_name}_{framework}_attempts.json"
        )
        attempt_records[m] = []

    jsonl_path = get_jsonl_path(framework=framework, benchmark_version=benchmark_version)
    states = build_task_states(jsonl_path, models, benchmark_version=benchmark_version)
    print(f"🚀 Loaded {len(states)} model-task states ({len(models)} models x tasks).")

    global_inputs = load_global_inputs(framework)
    v2_evaluator = (
        QiskitV2Evaluator(load_qiskit_v2_tasks(QISKIT_V2_JSONL), global_inputs)
        if benchmark_version == "v2"
        else None
    )

    for iteration in range(1, feedback_num + 1):
        pending = [
            s for s in states if (not s.done) and (s.attempts_used < feedback_num)
        ]
        if not pending:
            print("✅ No pending tasks left. Stopping early.")
            break

        print(f"\n----- Iteration {iteration}/{feedback_num} — pending: {len(pending)}")
        print("")
        print("Starting API requests...")

        reqs = build_requests_for_states(pending, prefill=prefill)
        raw_responses = send_requests_in_parallel(reqs, max_workers=16)

        for st, req, raw in zip(pending, reqs, raw_responses):
            assistant_text = extract_assistant_text(raw)

            parsed = None
            code = ""
            try:
                parsed = parse_response((raw, st.signature_prefill), st.entry_point)
                code = (
                    parsed.get("code")
                    or parsed.get("generated_code")
                    or parsed.get("content")
                    or assistant_text
                )
            except Exception:
                code = assistant_text

            st.attempts_used += 1
            eval_res = evaluate_generated_code(
                task_id=st.task_id,
                entry_point=st.entry_point,
                code=code,
                framework=framework,
                canonical_by_task=canonical_by_task,
                inputss=global_inputs,
                benchmark_version=benchmark_version,
                v2_evaluator=v2_evaluator,
            )

            feedback_to_model = ""
            if eval_res.kl_div_bool:
                st.done = True
            else:
                feedback_to_model = build_feedback_message(eval_res)
                st.history.append(
                    {
                        "attempt": st.attempts_used,
                        "assistant_code": code,
                        "feedback_to_model": feedback_to_model,
                    }
                )
                st.last_code = code
                st.last_feedback = feedback_to_model

            record = {
                "framework": framework,
                "model": st.model,
                "task_id": st.task_id,
                "entry_point": st.entry_point,
                "category": st.category,
                "attempt": st.attempts_used,
                "done_after_attempt": st.done,
                "request_payload": req,
                "raw_response": raw,
                "assistant_text": assistant_text,
                "parsed_response": parsed,
                "code": code,
                "evaluation": {
                    "compiled": eval_res.compiled,
                    "ran": eval_res.ran,
                    "kl_div_bool": eval_res.kl_div_bool,
                    "error": eval_res.error,
                    "output": None
                    if eval_res.output is None
                    else str(eval_res.output)[:2000],
                    "benchmark_version": benchmark_version,
                },
                "feedback_sent_to_model": feedback_to_model or None,
            }

            attempt_records[st.model].append(record)

        solved = sum(1 for s in states if s.done)
        print(f"📌 Progress after iteration {iteration}: solved {solved}/{len(states)}")

    for model in models:
        out_path = attempt_json_paths[model]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(attempt_records[model], f, indent=2, ensure_ascii=False)
        print(
            f"Saved attempts JSON: {out_path} ({len(attempt_records[model])} records)"
        )

    # Final summary per model
    for model in models:
        model_name = model.replace("/", "_")
        final = []
        for st in states:
            if st.model != model:
                continue
            final.append(
                {
                    "framework": framework,
                    "model": st.model,
                    "task_id": st.task_id,
                    "entry_point": st.entry_point,
                    "category": st.category,
                    "done": st.done,
                    "attempts_used": st.attempts_used,
                    "last_code": st.last_code,
                    "last_feedback": st.last_feedback,
                }
            )

        out_file = model_response_dir / f"{model_name}_{framework}_final.json"
        out_file.write_text(
            json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved final summary: {out_file}")
    print_feedback_accuracy_table(
        states=states,
        attempt_records=attempt_records,
        feedback_num=feedback_num,
        models=models,
    )
    print("🎉 Feedback loop complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feedback loop LLM evaluation on quantum tasks"
    )
    parser.add_argument("models", nargs="*", help="Model names to evaluate")
    parser.add_argument(
        "--framework",
        type=str,
        default="cirq",
        choices=["cirq", "pennylane", "qiskit"],
        help="Framework to use",
    )
    parser.add_argument(
        "--feedback_num", type=int, default=5, help="Max attempts per task"
    )
    parser.add_argument(
        "--benchmark-version",
        choices=["v1", "v2"],
        default="v1",
        help="Benchmark grader/prompt version (v2 currently qiskit only).",
    )
    args = parser.parse_args()

    models = args.models if args.models else DEFAULT_MODELS
    main(
        models=models,
        framework=args.framework,
        feedback_num=args.feedback_num,
        benchmark_version=args.benchmark_version,
    )
