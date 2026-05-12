from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re


CODING_RE = re.compile(r"^#.*coding[:=]\s*[-\w.]+")


def normalize_task_id(value: Any) -> str:
    task_id = str(value).strip()
    return task_id.zfill(2) if task_id.isdigit() and len(task_id) == 1 else task_id


def extract_prompt_header(complete_prompt: str) -> str:
    lines = complete_prompt.splitlines()
    start = _first_code_line(lines)
    if start is None:
        return ""

    header_lines: list[str] = []
    for line in lines[start:]:
        stripped = line.lstrip()
        if stripped.startswith("def "):
            break
        if stripped.startswith(("from ", "import ")):
            header_lines.append(line.rstrip())
        elif header_lines and line.strip() == "" and header_lines[-1] != "":
            header_lines.append("")

    while header_lines and header_lines[-1] == "":
        header_lines.pop()
    return "\n".join(header_lines)


def _first_code_line(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.lstrip().startswith(("from ", "import ", "def ")):
            return index
    return None


def load_prompts_jsonl_as_dict(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    prompts: dict[str, dict[str, Any]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_no} in {jsonl_path}: {exc}") from exc
            task_id = normalize_task_id(obj.get("task_id"))
            complete_prompt = obj.get("complete_prompt", "") or ""
            prompts[task_id] = {
                **obj,
                "task_id": task_id,
                "header": extract_prompt_header(complete_prompt),
            }
    return prompts


def code_has_import_header(code: str) -> bool:
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("import ", "from ")):
            return True
        if stripped.startswith(("def ", "class ")):
            return False
    return False


def add_header_if_missing(code: str, header: str) -> str:
    header = (header or "").strip()
    if not header or code_has_import_header(code):
        return code

    lines = code.splitlines(keepends=True)
    insert_at = _header_insert_index(lines)
    return "".join(lines[:insert_at]) + header + "\n\n" + "".join(lines[insert_at:])


def _header_insert_index(lines: list[str]) -> int:
    insert_at = 0
    if insert_at < len(lines) and lines[insert_at].startswith("#!"):
        insert_at += 1
    if insert_at < len(lines) and CODING_RE.match(lines[insert_at].rstrip("\n")):
        insert_at += 1
    while insert_at < len(lines) and (lines[insert_at].strip() == "" or lines[insert_at].lstrip().startswith("#")):
        insert_at += 1
    while insert_at < len(lines) and lines[insert_at].lstrip().startswith("from __future__ import"):
        insert_at += 1
    return insert_at
