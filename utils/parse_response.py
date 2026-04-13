import re


def extract_code_from_markdown(text: str, entry_point: str) -> str:
    """Extract Python code from markdown code blocks if present."""
    # Try to find ```python ... ``` blocks first
    python_blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if python_blocks:
        # Find the block containing the entry point function
        for block in python_blocks:
            if f"def {entry_point}" in block:
                return block.strip()
        # If no block has the entry point, return the first block
        return python_blocks[0].strip()

    # Try generic ``` ... ``` blocks
    generic_blocks = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if generic_blocks:
        for block in generic_blocks:
            if f"def {entry_point}" in block:
                return block.strip()
        return generic_blocks[0].strip()

    return text


def parse_response(args, entry_point):
    """Parse single response (first choice only)."""
    response = args[0]
    chat_completion = args[1]
    out = {}
    out["model"] = response.get("model")
    usage = response.get("usage") or {}

    out["usage"] = usage
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["completion_tokens"] = usage.get("completion_tokens")
    out["total_tokens"] = usage.get("total_tokens")
    cdet = usage.get("completion_tokens_details") or {}
    pdet = usage.get("prompt_tokens_details") or {}

    out["reasoning_tokens"] = cdet.get("reasoning_tokens")
    out["accepted_prediction_tokens"] = cdet.get("accepted_prediction_tokens")
    out["rejected_prediction_tokens"] = cdet.get("rejected_prediction_tokens")
    out["cached_tokens"] = pdet.get("cached_tokens")
    out["cache_write_tokens"] = pdet.get("cache_write_tokens")

    code = response.get("choices")[0].get("message").get("content")

    code = extract_code_from_markdown(code, entry_point)

    if ("def " + entry_point) not in code:
        print("This is a chat completion model!")
        code = chat_completion + code
    out["code"] = code
    return out

