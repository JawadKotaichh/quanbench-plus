from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
