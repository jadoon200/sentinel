.PHONY: env install lint typecheck test check up down ingest migrate

# One-time: create the conda env, then `conda activate sentinel`
env:
	conda create -y -n sentinel python=3.12

# Run inside the activated sentinel env
install:
	pip install -r requirements-dev.txt && pip install -e .

lint:
	ruff check . && ruff format --check .

typecheck:
	mypy

test:
	pytest

check: lint typecheck test

up:
	docker compose up -d db mlflow && docker compose run --rm migrate

down:
	docker compose down

migrate:
	alembic upgrade head

ingest:
	python -m sentinel.ingest.flows
