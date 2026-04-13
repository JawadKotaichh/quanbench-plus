from typing import Any, Dict


def parse_prompt(
    prompt: str,
    chat_completion: str,
    model: str,
    n: int = 1,
    temperature: float | None = None,
    prefill: bool = False,
):
    """
    Parse prompt into OpenRouter API request format.

    Args:
        prompt: The user prompt
        chat_completion: The assistant prefill content
        model: Model identifier
        n: Number of completions to generate (for pass@k)
        temperature: Sampling temperature. If None, uses 0.0 for n=1, 0.8 for n>1
    """
    out: Dict[str, Any] = {}
    messages = []
    messages.append({"role": "user", "content": prompt})
    if not prefill:
        chat_completion = ""

    messages.append({"role": "assistant", "content": chat_completion})
    out["messages"] = messages
    out["model"] = model
    out["stream"] = False
    out["reasoning"] = {"exclude": True}

    if temperature is not None:
        out["temperature"] = temperature
    else:
        out["temperature"] = 0.0 if n == 1 else 0.8

    if n > 1:
        out["n"] = n
    out["usage"] = {"include": True}
    # Force OpenRouter to use providers that support all parameters (chat completion)
    # out["provider"] = {"require_parameters": True}
    return out
