from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.get_function_signature_from_prompt import get_function_signature_from_prompt
from utils.read_jsonl import read_jsonl


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
    history: list[dict[str, Any]] = field(default_factory=list)
    last_code: str = ""
    last_feedback: str = ""


def build_task_states(jsonl_path: str, models: list[str]) -> list[TaskState]:
    states: list[TaskState] = []
    for model in models:
        for task in read_jsonl(jsonl_path):
            prompt = task.get("complete_prompt", "")
            states.append(
                TaskState(
                    task_id=str(task.get("task_id")),
                    entry_point=str(task.get("entry_point")),
                    category=str(task.get("category")),
                    model=model,
                    prompt=prompt,
                    signature_prefill=get_function_signature_from_prompt(prompt) or "",
                )
            )
    return states
