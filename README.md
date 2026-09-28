*This project has been created as part of the 42 curriculum by yel-ghaz.*

# Call Me Maybe

## Description

Call Me Maybe converts a natural language request into a structured function call. For example, a request to add 40 and 2 should produce the `fn_add_numbers` function with `a` set to `40` and `b` set to `2`; the program returns the call instead of calculating the sum.

The project uses the supplied `llm_sdk` package and the Qwen3-0.6B model. Available functions and test prompts are read from JSON files in `data/input/`. Results are written as a JSON array.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- Internet access on the first run so the SDK can download the model and tokenizer

### Install

From the project root, install the project dependencies:

```sh
make install
```

This runs `uv sync`. The SDK is included in `llm_sdk/` and is installed as a local dependency.

### Run

Use the supplied definitions and prompts:

```sh
make run
```

The results are written to `data/output/function_calling_results.json`.

Custom file paths can be passed to the module:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Other Makefile commands:

```sh
make debug  # Run with Python's debugger
make clean  # Remove local caches and generated output
```

## Algorithm explanation

The program first lists the available function names, descriptions, and parameter types in the model prompt. It then asks the model to select a function and identify values from the user's request.

Function-name decoding is constrained by the supplied function list. At every step, only vocabulary tokens that keep the generated text as a prefix of an available function name can be selected. This prevents the model from producing an unknown function name.

Parameter values are generated according to their declared types. Numeric and integer parameters allow only the corresponding numeric characters, boolean parameters are restricted to `true` or `false`, and strings are generated as printable text until a quote closes the value. The program assembles the call fields in a fixed order and serializes the results with Python's JSON encoder.

## Design decisions

- Use the supplied SDK and Qwen3-0.6B model, as expected by the assignment.
- Keep function definitions and prompts in separate JSON input files.
- Validate those files with small Pydantic models before model inference.
- Constrain the function name and parameter types during decoding; let Python's JSON encoder handle JSON escaping and serialization.
- Process prompts independently, so a generation failure for one prompt does not prevent the other prompts from producing results.

## Performance analysis

The model makes a new inference call for each generated token. As a result, runtime depends on the device, prompt length, and the number of tokens needed for the function name and parameter values. CPU inference is supported by the SDK, though it can take longer than an available GPU.

The constraints keep function names and parameter types within the supplied definitions. They do not guarantee that the model extracts the semantically correct value from every natural language request. The JSON output is serialized from Python dictionaries, so the written file is valid JSON even when an individual prompt fails; failed prompts are recorded with an empty name and parameter object.

## Challenges faced

- **Valid function names:** unrestricted generation can invent names. Prefix filtering limits each name token to choices that can complete a supplied function name.
- **Typed values:** different parameter types need different token rules. The decoder applies numeric, boolean, and printable-string rules based on each function definition.
- **Token text representation:** the vocabulary uses markers such as `Ġ` for spaces and `Ċ` for newlines. The decoder converts these markers before checking generated text.
- **Per-prompt failures:** model errors should not discard other results. The runner catches failures per prompt and records a result for each input.

## Testing strategy

All 11 supplied prompts were run through the model. The results used a defined function name, included its expected parameter keys, and had values of the declared parameter types. Edge-case checks covered empty files, malformed JSON, empty arrays, wrong input shapes, missing files, and a generation failure for one prompt; invalid inputs were rejected and processing continued after the individual generation failure. Flake8 and mypy were run with the project flags and reported no issues. Docstrings were checked for PEP 257 style.

## Example usage

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

For a prompt such as `What is the sum of 40 and 2?`, the output has this shape:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40.0,
    "b": 2.0
  }
}
```

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [Constrained beam search](https://huggingface.co/blog/constrained-beam-search)
- [uv documentation](https://docs.astral.sh/uv/)

### AI usage disclosure

AI assistance was used to organize the implementation into stages, draft and refine parts of the Python implementation. Understanding the constrained decoding and some concepts.
