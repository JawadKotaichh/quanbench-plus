from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import os
import time

import requests

from utils.coda_local import is_coda_model, send_coda_request
from utils.get_function_signature_from_prompt import get_function_signature_from_prompt
from utils.parse_prompt import parse_prompt
from utils.parse_response import parse_response
from utils.read_jsonl import read_jsonl


def send_request(request: dict[str, Any]) -> tuple[dict[str, Any], str]:
    messages = request.get("messages") or []
    chat_completion = messages[1].get("content", "") if len(messages) > 1 else ""
    if is_coda_model(request.get("model")):
        return send_coda_request(request), chat_completion

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=request,
        headers={"Authorization": f"Bearer {os.getenv('API_KEY')}", "Content-Type": "application/json"},
        timeout=180,
    )
    if response.status_code != 200:
        return {"error": {"status_code": response.status_code, "body": response.text}}, chat_completion
    return response.json(), chat_completion


def parse_requests(json_path: str, models: list[str], benchmark_version: str = "v1"):
    requests: list[dict[str, Any]] = []
    tasks_info: list[dict[str, Any]] = []
    for model in models:
        for task in read_jsonl(json_path):
            prompt = task.get("prompt_v2") if benchmark_version == "v2" else None
            prompt = prompt or task.get("complete_prompt")
            signature = get_function_signature_from_prompt(prompt)
            requests.append(parse_prompt(prompt, signature, model, n=1, prefill=False))
            tasks_info.append(
                {
                    "task_id": task.get("task_id"),
                    "entry_point": task.get("entry_point"),
                    "category": task.get("category"),
                    "model": model,
                }
            )
    return requests, tasks_info


def error_result(task_index: int, version: int, task_info: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "id": f"error-{int(time.time())}-{task_index}-v{version}",
        "choices": [
            {
                "finish_reason": "error",
                "native_finish_reason": "error",
                "message": {"role": "assistant", "content": f"Error: {exc}"},
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        **task_info,
        "version": version,
    }


def process_single_task_pass_k(args):
    task_index, request, task_info, version, pass_k = args
    single_request = {key: value for key, value in request.items() if key != "n"}
    single_request["temperature"] = 0.8 if pass_k > 1 else single_request.get("temperature", 0.8)
    try:
        raw_response = send_request(single_request)
        parsed_response = parse_response(raw_response, task_info.get("entry_point"))
        return task_index, version, {**task_info, **parsed_response, "version": version}
    except Exception as exc:
        print(f"Request task {task_info.get('task_id')} v{version} generated an exception: {exc}")
        return task_index, version, error_result(task_index, version, task_info, exc)


def pass_k_args(requests: list[dict[str, Any]], tasks_info: list[dict[str, Any]], pass_k: int):
    for task_index, (request, info) in enumerate(zip(requests, tasks_info)):
        for version in range(1, pass_k + 1):
            yield task_index, request, info, version, pass_k


def process_requests_pass_k(requests, tasks_info, pass_k):
    total_calls = len(requests) * pass_k
    print(f"   Sending {total_calls} requests with 16 concurrent workers (pass@{pass_k})...")
    results: list[dict[str, Any] | None] = [None] * total_calls
    all_args = list(pass_k_args(requests, tasks_info, pass_k))

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_single_task_pass_k, args): (args[0], args[3]) for args in all_args}
        for completed, future in enumerate(as_completed(futures), start=1):
            task_index, version = futures[future]
            out_index = task_index * pass_k + (version - 1)
            try:
                _, _, results[out_index] = future.result()
            except Exception as exc:
                results[out_index] = error_result(task_index, version, tasks_info[task_index], exc)
            if completed % 10 == 0:
                print(f"   Progress: {completed}/{total_calls} ({completed / total_calls * 100:.1f}%)")
            time.sleep(0.01)

    print(f"\n   Completed all {total_calls} requests (pass@{pass_k})")
    return [result for result in results if result is not None]
