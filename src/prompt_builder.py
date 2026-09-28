"""Build the context passed to the language model."""

from .models import FunctionDefinition


def build_prompt(
    user_request: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build model context from a request and available functions.

    Args:
        user_request: Natural language request to turn into a function call.
        functions: Function definitions available to the model.

    Returns:
        The formatted prompt text.

    """
    function_lines = []

    for function in functions:
        parameters = ", ".join(
            f"{name}: {definition.type}"
            for name, definition in function.parameters.items()
        )
        function_lines.append(
            f"- {function.name}: {function.description} "
            f"(parameters: {parameters})"
        )

    function_list = "\n".join(function_lines)
    return (
        "Convert the user request into a function call.\n"
        "Choose the most suitable function and identify its values.\n"
        "Available functions:\n"
        f"{function_list}\n\n"
        f"User request: {user_request}\n"
        "Function call:"
    )
