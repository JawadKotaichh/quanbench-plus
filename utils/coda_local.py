from __future__ import annotations

import json
import os
from typing import Any

import requests


CODA_MODEL_ALIASES = {"coda", "coda-local", "coda/local"}


def is_coda_model(model: str | None) -> bool:
    return (model or "").strip().lower() in CODA_MODEL_ALIASES


def _messages_from_payload(request_payload: dict[str, Any]) -> list[dict[str, str]]:
    messages = [
        {"role": msg["role"], "content": msg.get("content", "")}
        for msg in request_payload.get("messages", [])
        if msg.get("role") in {"user", "assistant"} and msg.get("content")
    ]
    if not messages:
        raise RuntimeError("Coda local request has no non-empty messages.")
    return messages


def _error_response(model: str, error_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "error": error_payload,
        "choices": [
            {
                "finish_reason": "error",
                "message": {"role": "assistant", "content": str(error_payload)},
            }
        ],
        "usage": {},
    }


def _stream_agent_response(response: requests.Response) -> tuple[list[str], dict[str, Any] | None, dict[str, Any] | None]:
    tokens: list[str] = []
    structured_response: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None = None

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload_text = line[len("data:") :].strip()
        if not payload_text:
            continue
        payload = json.loads(payload_text)
        payload_type = payload.get("type")
        if payload_type == "token":
            tokens.append(str(payload.get("content", "")))
        elif payload_type == "completed":
            structured_response = payload.get("structured_response") or {}
            break
        elif payload_type == "error":
            error_payload = payload
            break
    return tokens, structured_response, error_payload


def send_coda_request(request_payload: dict[str, Any]) -> dict[str, Any]:
    model = str(request_payload.get("model") or "coda-local")
    base_url = os.getenv("CODA_AGENTS_URL", "http://127.0.0.1:8000").rstrip("/")
    timeout = float(os.getenv("CODA_AGENTS_TIMEOUT", "900"))
    fast = os.getenv("CODA_AGENTS_FAST", "").lower() in {"1", "true", "yes"}
    response = requests.post(
        f"{base_url}/agents",
        json={"messages": _messages_from_payload(request_payload), "mode": "build", "fast": fast},
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()

    tokens, structured_response, error_payload = _stream_agent_response(response)
    if error_payload:
        return _error_response(model, error_payload)

    content = "".join(tokens)
    if structured_response:
        code = structured_response.get("code") or structured_response.get("content")
        if code:
            content = f"```python\n{code}\n```"

    return {
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "native_finish_reason": "completed",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {},
        "structured_response": structured_response,
    }
