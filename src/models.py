"""Data models for function definitions and prompts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ParameterDefinition(BaseModel):
    """Describe one function parameter and its supported JSON type."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "number", "integer", "boolean"]


class ReturnDefinition(BaseModel):
    """Describe the function's return value type."""

    model_config = ConfigDict(extra="forbid")

    type: str


class FunctionDefinition(BaseModel):
    """Describe a callable function and its parameters."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class Prompt(BaseModel):
    """Represent one natural language request from the input file."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
