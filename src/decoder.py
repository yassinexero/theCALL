"""Generate function calls using constrained token selection."""

import json
from typing import Any

import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .models import FunctionDefinition
from .prompt_builder import build_prompt


class FunctionCallGenerator:
    """Generate a function name and typed values using constrained tokens.

    Args:
        model: The supplied small language model.
        functions: Function definitions the model can select from.

    """

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[FunctionDefinition],
    ) -> None:
        """Store the model and load its token vocabulary."""
        self.model = model
        self.functions = functions

        vocab_path = model.get_path_to_vocab_file()
        with open(vocab_path, encoding="utf-8") as vocab_file:
            raw_vocab: dict[str, int] = json.load(vocab_file)

        self.vocabulary = {
            token_id: token.replace("Ġ", " ").replace("Ċ", "\n")
            for token, token_id in raw_vocab.items()
        }

    def _encode(self, text: str) -> list[int]:
        """Encode text into model token IDs."""
        token_ids = self.model.encode(text).flatten()
        return [int(token_id) for token_id in token_ids]

    def _choose_token(self, input_ids: list[int], allowed: list[int]) -> int:
        """Choose the highest-scoring token from the allowed IDs."""
        if not allowed:
            raise ValueError("No valid next tokens are available")

        logits = np.asarray(self.model.get_logits_from_input_ids(input_ids))
        masked_logits = np.full(logits.shape, -np.inf)
        valid_ids = [
            token_id for token_id in allowed if token_id < len(logits)
        ]
        if not valid_ids:
            raise ValueError("No valid token IDs are available in the model")
        for token_id in valid_ids:
            masked_logits[token_id] = logits[token_id]
        return int(np.argmax(masked_logits))

    def _generate_function_name(self, input_ids: list[int]) -> str:
        """Generate a function name while keeping it a valid name prefix."""
        names = [function.name for function in self.functions]
        result = ""

        for _ in range(max(map(len, names))):
            allowed = [
                token_id
                for token_id, token in self.vocabulary.items()
                if any(name.startswith(result + token) for name in names)
            ]
            if not allowed:
                break

            token_id = self._choose_token(input_ids, allowed)
            result += self.vocabulary[token_id]
            input_ids.append(token_id)
            if result in names:
                return result

        raise ValueError("The model could not choose a valid function name")

    def _generate_number(
        self,
        input_ids: list[int],
        integer: bool,
    ) -> int | float:
        """Generate a numeric value using only number characters."""
        allowed_chars = "0123456789-" if integer else "0123456789-."
        result = ""

        for _ in range(20):
            allowed = [
                token_id
                for token_id, token in self.vocabulary.items()
                if token.strip()
                and all(char in allowed_chars for char in token.strip())
                and ("-" not in token.strip() or not result)
                and (
                    integer
                    or token.strip().count(".") + result.count(".") <= 1
                )
            ]
            if result and any(char.isdigit() for char in result):
                allowed.extend(
                    token_id
                    for token_id, token in self.vocabulary.items()
                    if token.strip() in (",", "}")
                )

            token_id = self._choose_token(input_ids, allowed)
            token = self.vocabulary[token_id].strip()
            if token in (",", "}"):
                break
            result += token
            input_ids.append(token_id)

        try:
            return int(result) if integer else float(result)
        except ValueError as error:
            message = "The model did not generate a valid number"
            raise ValueError(message) from error

    def _generate_boolean(self, input_ids: list[int]) -> bool:
        """Generate a boolean value from the true and false literals."""
        choices: dict[str, bool] = {"true": True, "false": False}
        built = ""

        for _ in range(5):
            allowed = [
                token_id
                for token_id, token in self.vocabulary.items()
                if any(
                    value.startswith(built + token.strip().lower())
                    for value in choices
                )
            ]
            token_id = self._choose_token(input_ids, allowed)
            built += self.vocabulary[token_id].strip().lower()
            input_ids.append(token_id)
            if built in choices:
                return choices[built]

        raise ValueError("The model did not generate true or false")

    def _generate_string(self, input_ids: list[int]) -> str:
        """Generate printable text until the model emits a quote token."""
        result = ""

        for _ in range(50):
            logits = np.asarray(
                self.model.get_logits_from_input_ids(input_ids)
            )
            token_id = int(np.argmax(logits))
            token = self.vocabulary[token_id]
            if '"' in token:
                result += token.split('"', 1)[0]
                break
            if not token.isprintable():
                message = "The model generated a non-printable string token"
                raise ValueError(message)
            result += token
            input_ids.append(token_id)

        return result.strip()

    def generate(self, user_request: str) -> dict[str, Any]:
        """Generate a structured function call for one user request.

        Args:
            user_request: Natural language request to convert.

        Returns:
            A dictionary containing the prompt, function name, and parameters.

        """
        prompt = build_prompt(user_request, self.functions)
        input_ids = self._encode(prompt + ' {"name": "')
        function_name = self._generate_function_name(input_ids)
        input_ids.extend(self._encode('", "parameters": {'))

        function = next(
            item for item in self.functions if item.name == function_name
        )
        parameters: dict[str, Any] = {}
        for index, (name, definition) in enumerate(
            function.parameters.items()
        ):
            input_ids.extend(self._encode(f'"{name}": '))
            if definition.type == "number":
                parameter_value: Any = self._generate_number(
                    input_ids,
                    integer=False,
                )
            elif definition.type == "integer":
                parameter_value = self._generate_number(
                    input_ids,
                    integer=True,
                )
            elif definition.type == "boolean":
                parameter_value = self._generate_boolean(input_ids)
            else:
                input_ids.extend(self._encode('"'))
                parameter_value = self._generate_string(input_ids)
                input_ids.extend(self._encode('"'))

            parameters[name] = parameter_value
            if index + 1 < len(function.parameters):
                input_ids.extend(self._encode(", "))

        return {
            "prompt": user_request,
            "name": function_name,
            "parameters": parameters,
        }
