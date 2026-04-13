from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
from pass_at_k_pipeline.qiskit_pip.save_qiskit_responses import save_qiskit_responses
import json
import sys
from utils.get_kl_div import get_kl_div
from utils.get_canonical_results import GLOBAL_INPUTS
from pass_at_k_pipeline.defaults import DEFAULT_MODELS
from utils.evaluation_summary import print_evaluation_summary
from pass_at_k_pipeline.qiskit_pip.paths import (
    MODEL_RESPONSES_DIR,
    RESPONSES_OUTPUT_DIR,
    RESULTS_OUTPUT_DIR,
    CANONICAL_SOLUTIONS_DIR,
)
from qiskit.circuit import QuantumCircuit
import numpy as np


def task_6_input():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz((25 * np.pi) / 54, 0)
    return qc


GLOBAL_INPUTS["06"] = task_6_input()


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_list(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def build_paths(models):
    model_names = [name.replace("/", "_") for name in models]
    model_responses_paths = []
    responses_paths = []
    results_paths = []
    for model_name in model_names:
        current_path = MODEL_RESPONSES_DIR / f"{model_name}_qiskit.json"
        model_responses_paths.append((current_path, model_name))
        responses_paths.append(RESPONSES_OUTPUT_DIR / f"{model_name}.json")
        results_paths.append(RESULTS_OUTPUT_DIR / f"{model_name}.json")
    return model_responses_paths, responses_paths, results_paths


def group_responses_by_task(
    model_responses: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groups responses by task_id and keeps ALL versions.
    Sorts each task's list by 'version' (if present).
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for resp in model_responses:
        task_id = str(resp["task_id"])
        grouped[task_id].append(resp)

    # sort versions for deterministic order
    for task_id, lst in grouped.items():
        lst.sort(key=lambda r: int(r.get("version", 1)))

    return grouped


def extract_token_fields(resp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_tokens": resp.get("prompt_tokens"),
        "completion_tokens": resp.get("completion_tokens"),
        "total_tokens": resp.get("total_tokens"),
        "reasoning_tokens": resp.get("reasoning_tokens"),
        "accepted_prediction_tokens": resp.get("accepted_prediction_tokens"),
        "rejected_prediction_tokens": resp.get("rejected_prediction_tokens"),
        "cached_tokens": resp.get("cached_tokens"),
        "cache_write_tokens": resp.get("cache_write_tokens"),
    }


def main(
    model_responses_path: Path,
    model_name: str,
    response_path: Path,
    result_path: Path,
    canonical_solutions_path: Path,
    global_inputs,
) -> List[Dict[str, Any]]:
    save_qiskit_responses(
        file=model_responses_path,
        response_path=response_path,
        inputss=global_inputs,
    )
    model_responses = load_json_list(path=response_path)
    canonical_solutions = load_json_list(path=canonical_solutions_path)

    canonical_by_task: Dict[str, Dict[str, Any]] = {
        str(sol["task_id"]): sol for sol in canonical_solutions
    }
    responses_by_task = group_responses_by_task(model_responses)

    results_out: List[Dict[str, Any]] = []
    for task_id, versions in responses_by_task.items():
        canonical = canonical_by_task.get(task_id)

        record: Dict[str, Any] = {
            "task_id": task_id,
            "model_name": model_name,
            "category": versions[0].get("category", "missing")
            if versions
            else "missing",
            "pass_k": len(versions),
            "compiled_versions": 0,
            "passed_versions": 0,
            "any_compiled": False,
            "any_passed": False,
            "canonical_output": "no canonical output",
            "versions": [],  # per-version details
        }
        if canonical is None:
            for resp in versions:
                token_fields = extract_token_fields(resp)
                record["versions"].append(
                    {
                        "version": int(resp.get("version", 1)),
                        **token_fields,
                        "error": resp.get("error") or "no canonical solution",
                        "kl_value": None,
                        "kl_bool": False,
                        "response_output": [],
                    }
                )
            results_out.append(record)
            continue
        canonical_probs = canonical.get("canonical_output")
        if canonical_probs is None:
            for resp in versions:
                token_fields = extract_token_fields(resp)
                record["versions"].append(
                    {
                        "version": int(resp.get("version", 1)),
                        **token_fields,
                        "error": (resp.get("error") or "")
                        + " (and no canonical_output)",
                        "kl_value": None,
                        "kl_bool": False,
                        "response_output": [],
                    }
                )
            results_out.append(record)
            continue

        record["canonical_output"] = canonical_probs
        for resp in versions:
            token_fields = extract_token_fields(resp)
            v = int(resp.get("version", 1))
            err = resp.get("error")
            no_error = not bool(err)
            vrec: Dict[str, Any] = {
                "version": v,
                **token_fields,
                "error": err,
                "kl_value": None,
                "kl_bool": False,
                "response_output": [],
            }

            if not no_error:
                record["versions"].append(vrec)
                continue

            model_probs = resp.get("output")
            if model_probs is None:
                vrec["error"] = "missing payload"
                record["versions"].append(vrec)
                continue

            if len(model_probs) != len(canonical_probs):
                vrec["error"] = (
                    f"shape mismatch: model_probs len {len(model_probs)}, "
                    f"canonical_probs len {len(canonical_probs)}"
                )
                record["versions"].append(vrec)
                continue

            kl_value, kl_bool = get_kl_div(
                probs=model_probs,
                expected_probs=canonical_probs,
            )

            vrec["kl_value"] = float(kl_value)
            vrec["kl_bool"] = bool(kl_bool)
            vrec["response_output"] = model_probs
            vrec["error"] = None

            record["compiled_versions"] += 1
            if vrec["kl_bool"]:
                record["passed_versions"] += 1

            record["versions"].append(vrec)

        record["any_compiled"] = record["compiled_versions"] > 0
        record["any_passed"] = record["passed_versions"] > 0

        results_out.append(record)

    save_json_list(result_path, results_out)
    return results_out


def get_qiskit_results(models: List[str], passk: int) -> None:
    RESPONSES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_responses_paths, responses_paths, results_paths = build_paths(models=models)
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for model_resp_path, resp_path, result_path in zip(
        model_responses_paths, responses_paths, results_paths
    ):
        results = main(
            model_responses_path=model_resp_path[0],
            model_name=model_resp_path[1],
            response_path=resp_path,
            result_path=result_path,
            canonical_solutions_path=CANONICAL_SOLUTIONS_DIR,
            global_inputs=GLOBAL_INPUTS,
        )
        all_results[model_resp_path[1]] = results

    print_evaluation_summary(all_results, passk)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        models = DEFAULT_MODELS
        passk = 1
    else:
        passk = int(sys.argv[-1])
        models = sys.argv[1:-1]
        if not models:
            models = DEFAULT_MODELS

    get_qiskit_results(models, passk=passk)
