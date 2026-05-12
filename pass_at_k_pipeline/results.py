from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
import json

from graders.contracts import V2Evaluator
from utils.get_kl_div import get_kl_div


SaveResponsesFn = Callable[..., tuple[list[dict[str, Any]], str | None]]


def load_json_list(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_list(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def group_responses_by_task(model_responses: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in model_responses:
        grouped[str(response["task_id"])].append(response)
    for responses in grouped.values():
        responses.sort(key=lambda response: int(response.get("version", 1)))
    return grouped


def token_fields(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_tokens": response.get("prompt_tokens"),
        "completion_tokens": response.get("completion_tokens"),
        "total_tokens": response.get("total_tokens"),
        "reasoning_tokens": response.get("reasoning_tokens"),
        "accepted_prediction_tokens": response.get("accepted_prediction_tokens"),
        "rejected_prediction_tokens": response.get("rejected_prediction_tokens"),
        "cached_tokens": response.get("cached_tokens"),
        "cache_write_tokens": response.get("cache_write_tokens"),
    }


def empty_version_record(response: dict[str, Any], error: str | None) -> dict[str, Any]:
    return {
        "version": int(response.get("version", 1)),
        **token_fields(response),
        "error": error,
        "kl_value": None,
        "kl_bool": False,
        "response_output": [],
    }


def make_task_record(
    task_id: str,
    model_name: str,
    versions: list[dict[str, Any]],
    benchmark_version: str,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "model_name": model_name,
        "category": versions[0].get("category", "missing") if versions else "missing",
        "pass_k": len(versions),
        "compiled_versions": 0,
        "passed_versions": 0,
        "any_compiled": False,
        "any_passed": False,
        "canonical_output": "no canonical output",
        "benchmark_version": benchmark_version,
        "versions": [],
    }


def grade_v1(model_probs: Any, canonical_probs: Any) -> tuple[Any, bool, dict[str, Any] | None]:
    if len(model_probs) != len(canonical_probs):
        raise ValueError(
            f"shape mismatch: model_probs len {len(model_probs)}, "
            f"canonical_probs len {len(canonical_probs)}"
        )
    kl_value, passed = get_kl_div(probs=model_probs, expected_probs=canonical_probs)
    return kl_value, bool(passed), None


def grade_v2(
    *,
    response: dict[str, Any],
    task_id: str,
    v2_task: dict[str, Any],
    evaluator: V2Evaluator,
) -> tuple[Any, bool, list[float], dict[str, Any]]:
    execution, details = evaluator.grade_code(
        task_id=task_id,
        code=response.get("code") or "",
        entry_point=response.get("entry_point") or v2_task["entry_point"],
    )
    metric = details["kl_value"] if "kl_value" in details else details.get("metric")
    return metric, bool(details["passed"]), execution.probabilities, details


def successful_version_record(
    response: dict[str, Any],
    *,
    kl_value: Any,
    passed: bool,
    output: Any,
    grader_details: dict[str, Any] | None,
    v2_task: dict[str, Any] | None,
) -> dict[str, Any]:
    record = empty_version_record(response, error=None)
    record["kl_value"] = None if kl_value is None else float(kl_value)
    record["kl_bool"] = passed
    record["response_output"] = output
    if grader_details is not None and v2_task is not None:
        record["grader_type"] = grader_details.get("grader_type")
        record["grader_details"] = grader_details
        record["canonical_class"] = v2_task["canonical_class"]
    return record


def evaluate_version(
    response: dict[str, Any],
    *,
    task_id: str,
    canonical_probs: Any,
    benchmark_version: str,
    v2_tasks: dict[str, dict[str, Any]],
    v2_evaluator: V2Evaluator | None,
) -> dict[str, Any]:
    if response.get("error"):
        return empty_version_record(response, response.get("error"))
    model_probs = response.get("output")
    if model_probs is None:
        return empty_version_record(response, "missing payload")

    try:
        if benchmark_version == "v2":
            if v2_evaluator is None or task_id not in v2_tasks:
                return empty_version_record(response, "missing v2 evaluator or canonical_class")
            metric, passed, output, details = grade_v2(
                response=response,
                task_id=task_id,
                v2_task=v2_tasks[task_id],
                evaluator=v2_evaluator,
            )
            return successful_version_record(
                response,
                kl_value=metric,
                passed=passed,
                output=output,
                grader_details=details,
                v2_task=v2_tasks[task_id],
            )

        metric, passed, _ = grade_v1(model_probs, canonical_probs)
        return successful_version_record(
            response,
            kl_value=metric,
            passed=passed,
            output=model_probs,
            grader_details=None,
            v2_task=None,
        )
    except Exception as exc:
        return empty_version_record(response, f"grading failed: {type(exc).__name__}: {exc}")


def evaluate_model_responses(
    *,
    model_responses_path: Path,
    model_name: str,
    response_path: Path,
    result_path: Path,
    canonical_solutions_path: Path,
    global_inputs: dict[str, Any],
    benchmark_version: str,
    save_responses: SaveResponsesFn,
    v2_tasks: dict[str, dict[str, Any]],
    v2_evaluator: V2Evaluator | None,
) -> list[dict[str, Any]]:
    save_responses(
        file=model_responses_path,
        response_path=response_path,
        inputss=global_inputs,
        benchmark_version=benchmark_version,
    )
    canonical_by_task = {str(task["task_id"]): task for task in load_json_list(canonical_solutions_path)}
    responses_by_task = group_responses_by_task(load_json_list(response_path))

    results: list[dict[str, Any]] = []
    for task_id, versions in responses_by_task.items():
        record = make_task_record(task_id, model_name, versions, benchmark_version)
        canonical = canonical_by_task.get(task_id)
        if canonical is None or canonical.get("canonical_output") is None:
            error = "no canonical solution" if canonical is None else "no canonical_output"
            record["versions"] = [empty_version_record(response, error) for response in versions]
            results.append(record)
            continue

        record["canonical_output"] = canonical["canonical_output"]
        for response in versions:
            version_record = evaluate_version(
                response,
                task_id=task_id,
                canonical_probs=canonical["canonical_output"],
                benchmark_version=benchmark_version,
                v2_tasks=v2_tasks,
                v2_evaluator=v2_evaluator,
            )
            record["versions"].append(version_record)
            record["compiled_versions"] += int(version_record["error"] is None)
            record["passed_versions"] += int(version_record["kl_bool"])

        record["any_compiled"] = record["compiled_versions"] > 0
        record["any_passed"] = record["passed_versions"] > 0
        results.append(record)

    save_json_list(result_path, results)
    return results
