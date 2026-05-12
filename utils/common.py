from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import tempfile

from utils.prompt_headers import (
    add_header_if_missing,
    code_has_import_header,
    extract_prompt_header,
    load_prompts_jsonl_as_dict,
    normalize_task_id,
)
from utils.serialization import extract_token_fields, save_json, to_jsonable


_to_jsonable = to_jsonable
_extract_token_fields = extract_token_fields


def execute_code_with_args(
    code: str,
    entry_point: str,
    arg1: Any = None,
    arg2: Any = None,
    arg3: Any = None,
    use_sandbox: bool = True,
) -> Any:
    namespace: dict[str, Any] = {}
    old_cwd = os.getcwd()
    try:
        if use_sandbox:
            os.chdir(Path(tempfile.mkdtemp(prefix="model_task_sandbox_")))
        try:
            exec(code, namespace, namespace)
        except SyntaxError as exc:
            print(f"\n[SyntaxError] line={exc.lineno}, msg={exc.msg}")
            print("----- CODE CAUSING ERROR -----")
            print(code)
            print("------------------------------")
            raise
        function = namespace.get(entry_point)
        if not callable(function):
            raise RuntimeError(f"Entry point '{entry_point}' not found or not callable.")
        args = [value for value in (arg1, arg2, arg3) if value is not None]
        return function(*args)
    finally:
        os.chdir(old_cwd)


def handle_task_04(code: str, entry_point: str, graph: Any, betta: Any, gamma: Any) -> Any:
    return execute_code_with_args(code, entry_point, graph, betta, gamma)


def handle_task_06(code: str, entry_point: str, gate_list: Any) -> Any:
    return execute_code_with_args(code, entry_point, gate_list)


def handle_task_29(code: str, entry_point: str, alice: Any, bob: Any) -> Any:
    return execute_code_with_args(code, entry_point, alice, bob)


def handle_task_39(code: str, entry_point: str, array: Any) -> Any:
    return execute_code_with_args(code, entry_point, array)


def handle_task_40(code: str, entry_point: str, params: Any) -> Any:
    return execute_code_with_args(code, entry_point, params)


def handle_task_41(code: str, entry_point: str, params: Any) -> Any:
    return execute_code_with_args(code, entry_point, params)


def handle_task_42(code: str, entry_point: str, theta: Any, phi: Any, lam: Any) -> Any:
    return execute_code_with_args(code, entry_point, theta, phi, lam)


def get_handler(task_id: str, code: str, entry_point: str, inputs: dict[str, Any]) -> Any:
    handlers = {
        "04": lambda: handle_task_04(code, entry_point, inputs[task_id][0], inputs[task_id][1], inputs[task_id][2]),
        "06": lambda: handle_task_06(code, entry_point, inputs[task_id]),
        "29": lambda: handle_task_29(code, entry_point, inputs[task_id][0], inputs[task_id][1]),
        "39": lambda: handle_task_39(code, entry_point, inputs[task_id]),
        "40": lambda: handle_task_40(code, entry_point, inputs[task_id]),
        "41": lambda: handle_task_41(code, entry_point, inputs[task_id]),
        "42": lambda: handle_task_42(code, entry_point, inputs[task_id][0], inputs[task_id][1], inputs[task_id][2]),
    }
    return handlers.get(task_id, lambda: execute_code_with_args(code, entry_point))()


__all__ = [
    "_extract_token_fields",
    "_to_jsonable",
    "add_header_if_missing",
    "code_has_import_header",
    "execute_code_with_args",
    "extract_prompt_header",
    "get_handler",
    "load_prompts_jsonl_as_dict",
    "normalize_task_id",
    "save_json",
]
