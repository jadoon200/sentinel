.PHONY: env install lock lint typecheck test check up down ingest enrich train train-anomaly replay ids-spectral ids-beacon eval-beacon-ctu13 sqli waf-replay api ui briefing refresh eval-ensemble eval-cross eval-domain eval-cross-family eval-label-efficiency eval-conformal eval-conformal-cross migrate

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

# Spectral beacon detector — documented C2-channel ranking (honest negative)
ids-spectral:
	python -m sentinel.ids.spectral

# Beacon detector by data-size dispersion — the C2 signature periodicity missed
ids-beacon:
	python -m sentinel.ids.beacon

# Validate the beacon-dispersion detector on CTU-13 (many botnet families/channels)
eval-beacon-ctu13:
	python scripts/eval_beacon_ctu13.py

# Application-layer SQLi detector (payload char n-grams), cross-corpus eval
sqli:
	python -m sentinel.ids.sqli

# WAF replay: score HTTP requests through the SQLi detector into T1190 alerts
waf-replay:
	python -m sentinel.ids.waf_replay

# Replay Thu-Fri flows through both models into ATT&CK-tagged alerts (needs make up)
replay:
	python -m sentinel.ids.replay

# Cross-family few-shot study: the cross-network fix (downloads 2018 days)
eval-cross-family:
	python scripts/eval_cross_family.py

# Label-efficiency curve: how few target labels suffice, random vs active selection
eval-label-efficiency:
	python scripts/eval_label_efficiency.py

# Domain-adaptation study: can we beat the 2017->2018 transfer failure?
eval-domain:
	python scripts/eval_domain_adapt.py

# Ensemble coverage: per-family best detector + recall across the five detectors
eval-ensemble:
	python scripts/eval_ensemble.py

# Cross-dataset generalization: train 2017, test 2018 (downloads a 2018 day)
eval-cross:
	python scripts/eval_cross_dataset.py

# Threshold-policy shoot-out on the temporal split: static p99 vs conformal vs
# the label-free budget controller (the drift-robust operating point)
eval-conformal:
	python scripts/eval_conformal.py

# Does label-free recalibration recover detection across networks? (2017->2018)
eval-conformal-cross:
	python scripts/eval_conformal_cross.py

# Print the auto-generated daily threat briefing (needs make up + make api)
briefing:
	curl -s localhost:8000/briefing

# Full graph refresh: ingest -> enrich -> replay (cron/launchd-friendly wrapper)
refresh:
	./scripts/refresh.sh

# Serve the read-only knowledge-graph API on :8000 (needs make up)
api:
	uvicorn sentinel.api.app:app --reload

# React dashboard dev server on :5173 (needs make api in another shell)
ui:
	npm --prefix frontend install && npm --prefix frontend run dev
