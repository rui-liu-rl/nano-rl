.PHONY: install lint format test hooks

install:  ## Install the package with dev + viz extras
	uv pip install -e ".[dev,viz]"

hooks:  ## Install the pre-commit git hooks
	uv run pre-commit install

lint:  ## Check linting and formatting (no changes)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint issues and format the code
	uv run ruff check --fix .
	uv run ruff format .

test:  ## Run the CPU smoke tests
	uv run python tests/test_smoke.py
