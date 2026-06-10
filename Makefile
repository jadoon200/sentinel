.PHONY: install lint typecheck test check up down ingest migrate

install:
	uv sync

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

up:
	docker compose up -d db mlflow && docker compose run --rm migrate

down:
	docker compose down

migrate:
	uv run alembic upgrade head

ingest:
	uv run python -m sentinel.ingest.flows
