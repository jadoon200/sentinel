.PHONY: env install lock lint typecheck test check up down ingest enrich migrate

# One-time: create the conda env, then `conda activate sentinel`
env:
	conda create -y -n sentinel python=3.12

# Run inside the activated sentinel env
install:
	pip install -r requirements-dev.txt && pip install -e .

# Refreeze the pinned lock (CI and Docker install from it)
lock:
	printf -- '--extra-index-url https://download.pytorch.org/whl/cpu\n\n' > requirements.lock
	pip freeze --exclude-editable >> requirements.lock

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

# Tag ingested reports with ATT&CK techniques (downloads models on first run)
enrich:
	python -m sentinel.ingest.flows enrich
