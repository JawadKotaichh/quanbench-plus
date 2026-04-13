from typing import Any, Dict, List, Optional


def parse_prompt(
    prompt: str,
    chat_completion: str,
    model: str,
    n: int = 1,
    temperature: float | None = None,
    prefill: bool = False,
    extra_messages: Optional[List[Dict[str, str]]] = None,
):
    """
    Build an OpenRouter chat/completions request.
    - If prefill=True: add an assistant message with `chat_completion` to steer the output.
    - If prefill=False: do NOT add an empty assistant message.
    - If extra_messages is provided: it is appended after the initial user prompt (and before prefill).
    """
    out: Dict[str, Any] = {}
    messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]

    if extra_messages:
        messages.extend(extra_messages)

    if prefill:
        messages.append({"role": "assistant", "content": chat_completion})

    out["messages"] = messages
    out["model"] = model
    out["stream"] = False
    out["reasoning"] = {"exclude": True}

    if temperature is not None:
        out["temperature"] = temperature
    else:
        out["temperature"] = 0.8

    if n > 1:
        out["n"] = n

    return out
