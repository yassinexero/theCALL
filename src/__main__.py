"""Command-line entry point for the function calling application."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .decoder import FunctionCallGenerator
from .parser import load_function_definitions, load_prompts
from .runner import process_prompts


def main() -> int:
    """Run the function calling application from the command line."""
    parser = argparse.ArgumentParser(
        description="Generate function calls from natural language prompts."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path where the results JSON file will be written.",
    )
    args = parser.parse_args()

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompts(args.input)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    try:
        generator = FunctionCallGenerator(Small_LLM_Model(), functions)
    except Exception as error:
        print(
            f"Could not initialize the language model: {error}",
            file=sys.stderr,
        )
        return 1

    results: list[dict[str, Any]] = process_prompts(prompts, generator)
    output_path = Path(args.output)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        print(
            f"Could not write results to {args.output}: {error}",
            file=sys.stderr,
        )
        return 1

    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
