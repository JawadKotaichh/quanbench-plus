import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict

import numpy as np
CODING_RE = re.compile(r"^#.*coding[:=]\s*[-\w.]+")


def normalize_task_id(x: Any) -> str:
    """Normalize task ids like 1 -> '01', '04' -> '04'."""
    s = str(x).strip()
    return s.zfill(2) if s.isdigit() and len(s) == 1 else s

def extract_prompt_header(complete_prompt: str) -> str:
    """
    Extract the 'header' from the prompt: the import block that appears before the first 'def'.
    Example in your jsonl:
      I need you to complete...
      from qiskit import ...
      def foo(...):
          ...
    -> returns: 'from qiskit import ...'
    """
    lines = complete_prompt.splitlines()

    # Find where the code starts (first import/def)
    start = None
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith(("from ", "import ", "def ")):
            start = i
            break
    if start is None:
        return ""

    header_lines: list[str] = []
    for ln in lines[start:]:
        s = ln.lstrip()

        if s.startswith("def "):
            break

        if s.startswith(("from ", "import ")):
            header_lines.append(ln.rstrip())
        elif header_lines and ln.strip() == "":
            if header_lines[-1] != "":
                header_lines.append("")

    # Trim trailing blanks
    while header_lines and header_lines[-1] == "":
        header_lines.pop()

    return "\n".join(header_lines)


def load_prompts_jsonl_as_dict(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read the prompt jsonl and return a dict keyed by task_id:
      prompts['04']['header'] -> import header from the prompt
      prompts['04']['complete_prompt'] -> full prompt
      prompts['04']['entry_point'] -> entry point
      ... plus any other fields in each json line
    """
    prompts: Dict[str, Dict[str, Any]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Bad JSON on line {line_no} in {jsonl_path}: {e}"
                ) from e

            tid = normalize_task_id(obj.get("task_id"))
            complete_prompt = obj.get("complete_prompt", "") or ""

            prompts[tid] = {
                **obj,
                "task_id": tid,
                "header": extract_prompt_header(complete_prompt),
            }

    return prompts


def code_has_import_header(code: str) -> bool:
    """
    True if there's any import before the first def/class.
    (If model starts directly with `def ...`, returns False.)
    """
    for ln in code.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(("import ", "from ")):
            return True
        if s.startswith(("def ", "class ")):
            return False
    return False


def add_header_if_missing(code: str, header: str) -> str:
    """
    If `code` has no import header before the first def/class, insert the prompt header.
    Safe around shebang / encoding / __future__ imports.
    """
    header = (header or "").strip()
    if not header:
        return code

    if code_has_import_header(code):
        return code

    lines = code.splitlines(keepends=True)
    insert_at = 0

    if insert_at < len(lines) and lines[insert_at].startswith("#!"):
        insert_at += 1

    # Encoding comment (must be 1st or 2nd line)
    if insert_at < len(lines) and CODING_RE.match(lines[insert_at].rstrip("\n")):
        insert_at += 1

    # Skip blank/comment lines
    while insert_at < len(lines) and (
        lines[insert_at].strip() == "" or lines[insert_at].lstrip().startswith("#")
    ):
        insert_at += 1

    while insert_at < len(lines) and lines[insert_at].lstrip().startswith(
        "from __future__ import"
    ):
        insert_at += 1

    header_block = header + "\n\n"
    return "".join(lines[:insert_at]) + header_block + "".join(lines[insert_at:])


def execute_code_with_args(
    code: str,
    entry_point: str,
    arg1=None,
    arg2=None,
    arg3=None,
    use_sandbox: bool = True,
):
    """
    Execute the code and run the entry point safely with up to three arguments.
    If use_sandbox=True:
      - Create a temporary directory.
      - Change the working directory to that sandbox.
      - exec() the code there.
      - Call the entry point.
      - Restore the original working directory.
    """
    ns: dict[str, Any] = {}
    sandbox_dir = None
    old_cwd = os.getcwd()

    try:
        if use_sandbox:
            sandbox_dir = Path(tempfile.mkdtemp(prefix="model_task_sandbox_"))
            os.chdir(sandbox_dir)

        # Execute the provided code in an isolated namespace
        try:
            exec(code, ns, ns)
        except SyntaxError as e:
            print(f"\n[SyntaxError] line={e.lineno}, msg={e.msg}")
            print("----- CODE CAUSING ERROR -----")
            print(code)
            print("------------------------------")
            raise

        func = ns.get(entry_point)
        if not callable(func):
            raise RuntimeError(
                f"Entry point '{entry_point}' not found or not callable."
            )

        args = [x for x in (arg1, arg2, arg3) if x is not None]
        result = func(*args)
        return result

    finally:
        # Always restore original working directory
        os.chdir(old_cwd)
        # if sandbox_dir is not None and sandbox_dir.exists():
        #     shutil.rmtree(sandbox_dir)


def handle_task_04(code, entry_point, graph, betta, gamma):
    return execute_code_with_args(code, entry_point, graph, betta, gamma)


def handle_task_06(code, entry_point, gate_list):
    return execute_code_with_args(code, entry_point, gate_list)


def handle_task_29(code, entry_point, alice, bob):
    return execute_code_with_args(code, entry_point, alice, bob)


def handle_task_39(code, entry_point, array):
    return execute_code_with_args(code, entry_point, array)


def handle_task_40(code, entry_point, params):
    return execute_code_with_args(code, entry_point, params)


def handle_task_41(code, entry_point, params):
    return execute_code_with_args(code, entry_point, params)


def handle_task_42(code, entry_point, theta, phi, lam):
    return execute_code_with_args(code, entry_point, theta, phi, lam)


def get_handler(x, code: str, entry_point: str, inputs: dict):
    handlers = {
        "04": lambda: handle_task_04(
            code, entry_point, inputs[x][0], inputs[x][1], inputs[x][2]
        ),
        "06": lambda: handle_task_06(code, entry_point, inputs[x]),
        "29": lambda: handle_task_29(code, entry_point, inputs[x][0], inputs[x][1]),
        "39": lambda: handle_task_39(code, entry_point, inputs[x]),
        "40": lambda: handle_task_40(code, entry_point, inputs[x]),
        "41": lambda: handle_task_41(code, entry_point, inputs[x]),
        "42": lambda: handle_task_42(
            code, entry_point, inputs[x][0], inputs[x][1], inputs[x][2]
        ),
    }

    func = handlers.get(x, lambda: execute_code_with_args(code, entry_point))
    return func()


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj

def _extract_token_fields(task):
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

def save_json(records, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return str(out_path.resolve())
