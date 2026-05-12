from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    return obj


def extract_token_fields(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_tokens": task.get("prompt_tokens"),
        "completion_tokens": task.get("completion_tokens"),
        "total_tokens": task.get("total_tokens"),
        "reasoning_tokens": task.get("reasoning_tokens"),
        "accepted_prediction_tokens": task.get("accepted_prediction_tokens"),
        "rejected_prediction_tokens": task.get("rejected_prediction_tokens"),
        "cached_tokens": task.get("cached_tokens"),
        "cache_write_tokens": task.get("cache_write_tokens"),
    }


def save_json(records: list[dict[str, Any]], out_path: Path | str) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    return str(out_path.resolve())
