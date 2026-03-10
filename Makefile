.PHONY: install install-dev data sft dpo eval demo test lint format typecheck clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

## ── Data ─────────────────────────────────────────────────────────────────────

data:
	python scripts/download_data.py

data-small:
	python scripts/download_data.py --max-samples 5000

## ── Training ─────────────────────────────────────────────────────────────────

sft:
	python scripts/train_sft.py

dpo:
	python scripts/train_dpo.py

## ── Evaluation ───────────────────────────────────────────────────────────────

eval:
	python scripts/evaluate.py

demo:
	python scripts/demo.py --adapter checkpoints/dpo

## ── Tests ────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

## ── Code quality ─────────────────────────────────────────────────────────────

lint:
	ruff check src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

typecheck:
	mypy src/

## ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage
