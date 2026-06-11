.PHONY: env install lock lint typecheck test check up down ingest enrich train train-anomaly replay api migrate

# One-time: create the conda env, then `conda activate sentinel`
env:
	conda create -y -n sentinel python=3.12

# Run inside the activated sentinel env
install:
	pip install -r requirements-dev.txt && pip install -e .
	@[ "$$(uname)" = "Darwin" ] && pip install -r requirements-mlx.txt || true

# Refreeze the pinned lock (CI and Docker install from it)
lock:
	printf -- '--extra-index-url https://download.pytorch.org/whl/cpu\n\n' > requirements.lock
	pip freeze --exclude-editable --exclude mlx --exclude mlx-metal >> requirements.lock

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

# Train the IDS baseline on corrected CIC-IDS2017 (data/cicids2017/)
train:
	python -m sentinel.ids.train

# Train the benign-only autoencoder anomaly detector (temporal split)
train-anomaly:
	python -m sentinel.ids.anomaly

# Sequence-level anomaly model over per-host flow streams (MLX, experimental)
train-sequence:
	python -m sentinel.ids.sequence

# Host-profile fan-out detector (per-window cardinality stats, no NN)
train-profile:
	python -m sentinel.ids.profile

# Replay Thu-Fri flows through both models into ATT&CK-tagged alerts (needs make up)
replay:
	python -m sentinel.ids.replay

# Serve the read-only knowledge-graph API on :8000 (needs make up)
api:
	uvicorn sentinel.api.app:app --reload
