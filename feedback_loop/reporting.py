from __future__ import annotations

from typing import Any


def print_feedback_accuracy_table(
    states: list[Any],
    attempt_records: dict[str, list[dict[str, Any]]],
    feedback_num: int,
    models: list[str],
) -> None:
    print("\n" + "=" * 85)
    print("FEEDBACK RESULTS TABLE")
    print("=" * 85)
    for model in models:
        model_states = [state for state in states if state.model == model]
        total_tasks = len(model_states)
        if total_tasks == 0:
            continue
        per_task = _first_success_levels(model_states, attempt_records.get(model, []))
        _print_model_table(model, feedback_num, total_tasks, per_task)
    print("=" * 85 + "\n")


def _first_success_levels(states: list[Any], attempts: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    per_task = {
        state.task_id: {"first_compiled_level": -1, "first_accurate_level": -1}
        for state in states
    }
    for record in attempts:
        task_id = str(record.get("task_id", ""))
        attempt = int(record.get("attempt", 0))
        evaluation = record.get("evaluation", {}) or {}
        if task_id not in per_task or attempt <= 0:
            continue
        level = attempt - 1
        if evaluation.get("compiled") and per_task[task_id]["first_compiled_level"] == -1:
            per_task[task_id]["first_compiled_level"] = level
        if evaluation.get("kl_div_bool") and per_task[task_id]["first_accurate_level"] == -1:
            per_task[task_id]["first_accurate_level"] = level
    return per_task


def _print_model_table(model: str, feedback_num: int, total_tasks: int, per_task: dict[str, dict[str, int]]) -> None:
    print(f"\nModel: {model}")
    print("-" * 85)
    print(f"{'Feedback Level':<20} {'Compiled':<25} {'Accurate (Pass)':<25}")
    print("-" * 85)
    for level in range(feedback_num):
        compiled = sum(
            1
            for value in per_task.values()
            if value["first_compiled_level"] != -1 and value["first_compiled_level"] <= level
        )
        accurate = sum(
            1
            for value in per_task.values()
            if value["first_accurate_level"] != -1 and value["first_accurate_level"] <= level
        )
        label = f"{level} feedback" if level > 0 else "0 (no feedback)"
        print(f"{label:<20} {compiled}/{total_tasks} ({compiled / total_tasks * 100:.1f}%){'':<10} {accurate}/{total_tasks} ({accurate / total_tasks * 100:.1f}%)")
    print("-" * 85)
