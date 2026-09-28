"""Load and validate the project's JSON input files."""

from pathlib import Path
from typing import TypeVar, cast

from pydantic import TypeAdapter, ValidationError

from .models import FunctionDefinition, Prompt


T = TypeVar("T")


def _read_contents(path: str) -> str:
    """Read a non-empty UTF-8 JSON file."""
    file_path = Path(path)

    try:
        contents = file_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Cannot read {path}: {error}") from error

    if not contents.strip():
        raise ValueError(f"Input file is empty: {path}")

    return contents


def _parse_list(
    contents: str,
    path: str,
    adapter: TypeAdapter[list[T]],
) -> list[T]:
    """Parse a JSON array and validate each entry against a model."""
    try:
        values = adapter.validate_json(contents)
    except ValidationError as error:
        raise ValueError(f"Invalid data in {path}: {error}") from error

    if not values:
        raise ValueError(f"Input file contains no entries: {path}")
    return cast(list[T], values)


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    contents = _read_contents(path)
    adapter: TypeAdapter[list[FunctionDefinition]] = TypeAdapter(
        list[FunctionDefinition]
    )
    return _parse_list(contents, path, adapter)


def load_prompts(path: str) -> list[Prompt]:
    """Load and validate prompts from a JSON file."""
    contents = _read_contents(path)
    adapter: TypeAdapter[list[Prompt]] = TypeAdapter(list[Prompt])
    return _parse_list(contents, path, adapter)
