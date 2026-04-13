import json


def read_jsonl(file_path):
    """Read JSONL file and return list of dictionaries."""
    out = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)
                out.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}")
                print(f"JSON decode error: {e}")
                return
    return out
