.PHONY: install run debug clean lint


UV_CACHE="$(HOME)/goinfre/.uv_cache"
export HF_HOME=$(HOME)/goinfre/.hf_cache

install:
	uv sync --cache-dir=$(UV_CACHE)

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --follow-imports=silent

clean:
	rm -rf .mypy_cache __pycache__ src/__pycache__ data/output
