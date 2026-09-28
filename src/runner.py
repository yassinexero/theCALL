"""Process prompts and collect generated function calls."""

import sys
from typing import Any

from .decoder import FunctionCallGenerator
from .models import Prompt


def process_prompts(
    prompts: list[Prompt],
    generator: FunctionCallGenerator,
) -> list[dict[str, Any]]:
    """Generate results for all prompts, continuing after individual errors.

    Args:
        prompts: Validated prompts to process.
        generator: Function call generator used for each request.

    Returns:
        One result dictionary for each input prompt.

    """
    results = []

    for item in prompts:
        try:
            results.append(generator.generate(item.prompt))
        except Exception as error:
            print(
                f"Could not process prompt '{item.prompt}': {error}",
                file=sys.stderr,
            )
            results.append({
                "prompt": item.prompt,
                "name": "",
                "parameters": {},
            })

    return results
