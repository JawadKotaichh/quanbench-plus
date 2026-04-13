def get_function_signature_from_prompt(prompt: str) -> str:
    """This is for the chat completion: extract function signature from the prompt"""
    return "\n".join(prompt.split("\n")[1:])
