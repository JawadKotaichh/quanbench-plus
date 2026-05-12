from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import os
import time

import requests
from dotenv import load_dotenv

from feedback_loop.v1_state import TaskState
from utils.coda_local import is_coda_model, send_coda_request
from utils.parse_prompt_with_feedback import parse_prompt

load_dotenv()


def extract_assistant_text(raw: dict[str, Any]) -> str:
    try:
        return raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def send_request(request_payload: dict[str, Any]) -> dict[str, Any]:
    if is_coda_model(request_payload.get("model")):
        return send_coda_request(request_payload)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY in environment.")
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=request_payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=180,
    )
    return response.json()


def send_requests_in_parallel(request_payloads: list[dict[str, Any]], max_workers: int = 16) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(request_payloads)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(send_request, payload): index for index, payload in enumerate(request_payloads)}
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                results[index] = _error_response(index, exc)
            if completed % 10 == 0 or completed == len(request_payloads):
                print(f"   Progress: {completed}/{len(request_payloads)}")
            time.sleep(0.05)
    return [result or {} for result in results]


def _error_response(index: int, exc: Exception) -> dict[str, Any]:
    return {
        "id": f"error-{int(time.time())}-{index}",
        "choices": [
            {
                "finish_reason": "error",
                "native_finish_reason": "error",
                "message": {"role": "assistant", "content": f"Error: {exc}"},
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def build_requests_for_states(states: list[TaskState], prefill: bool = False) -> list[dict[str, Any]]:
    return [
        parse_prompt(
            prompt=state.prompt,
            chat_completion=state.signature_prefill,
            model=state.model,
            n=1,
            prefill=prefill,
            extra_messages=_history_messages(state) or None,
        )
        for state in states
    ]


def _history_messages(state: TaskState) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in state.history:
        messages.append({"role": "assistant", "content": turn["assistant_code"]})
        messages.append({"role": "user", "content": turn["feedback_to_model"]})
    return messages
