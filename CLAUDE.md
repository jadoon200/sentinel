# SENTINEL

Cyber threat intelligence fusion platform: OSINT ingestion (NVD, CISA KEV, ATT&CK, OTX, RSS) + ML intrusion detection (CIC-IDS2017) + ATT&CK-based correlation. Portfolio project targeting Singapore's DIS Digital Work-Learn Scheme — deadline ~10 Aug 2026. Roadmap: `docs/ROADMAP.md`.

## Commands

- `uv sync` — install deps (Python 3.12+, uv-managed venv)
- `make check` — ruff lint + format check, mypy (strict), pytest. Run before every commit.
- `make up` / `make down` — Postgres + MLflow via Docker Compose, then Alembic migration
- `make ingest` — run the OSINT ingestion Prefect flow locally
- `uv run alembic revision -m "..."` / `upgrade head` — migrations live in `migrations/versions/`

## Layout

- `src/sentinel/config.py` — pydantic-settings; all config via `SENTINEL_*` env vars / `.env`
- `src/sentinel/db/` — SQLAlchemy 2.0 models (`models.py`), session helpers (`base.py`)
- `src/sentinel/ingest/` — one module per source (`nvd.py`, `kev.py`), Prefect flows in `flows.py`
- `tests/` — pytest; HTTP mocked with respx; DB tests on in-memory SQLite (models use `JSON().with_variant(JSONB, "postgresql")` to stay SQLite-compatible)

## Conventions

- Branching: `main` is protected/deployable; work on `feat/<topic>` branches, merge via PR once the GitHub repo has CI.
- mypy strict everywhere; Prefect decorator exception is scoped in `pyproject.toml` — don't widen it.
- Every ingester: httpx + tenacity retry, parse into ORM objects, upsert with `session.merge`, unit test with mocked payload.
- Zero-cost rule: free data sources and free/local models only. Claude API usage must stay behind an optional settings flag, never required.
- Large artifacts (datasets in `data/`, models, mlruns) never go in git.
