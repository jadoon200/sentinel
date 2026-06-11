# SENTINEL

Cyber threat intelligence fusion platform: OSINT ingestion (NVD, CISA KEV, ATT&CK, OTX, RSS) + ML intrusion detection (CIC-IDS2017) + ATT&CK-based correlation. Portfolio project targeting Singapore's DIS Digital Work-Learn Scheme — deadline ~10 Aug 2026. Roadmap: `docs/ROADMAP.md`.

## Commands

- Conda env `sentinel` (Python 3.12): `make env` once, then `conda activate sentinel` + `make install` (pip, `requirements*.txt`). Deps are declared in `requirements.txt` / `requirements-dev.txt` — keep `pyproject.toml` dependency lists in sync, and refreeze the pinned lock after changes with `make lock` (CI and Docker install from the lock).
- `make check` — ruff lint + format check, mypy (strict), pytest. Run before every commit, inside the activated env.
- `make up` / `make down` — Postgres + MLflow via Docker Compose, then Alembic migration
- `make ingest` — run the OSINT ingestion Prefect flow locally
- `alembic revision -m "..."` / `alembic upgrade head` — migrations live in `migrations/versions/`

## Layout

- `src/sentinel/config.py` — pydantic-settings; all config via `SENTINEL_*` env vars / `.env`
- `src/sentinel/db/` — SQLAlchemy 2.0 models (`models.py`), session helpers (`base.py`)
- `src/sentinel/ingest/` — one module per source (`nvd.py`, `kev.py`), Prefect flows in `flows.py`
- `tests/` — pytest; HTTP mocked with respx; DB tests on in-memory SQLite (models use `JSON().with_variant(JSONB, "postgresql")` to stay SQLite-compatible)

## Workflow

- Models: coding/implementation runs on Fable 5; documentation & context passes (CLAUDE.md, README, ROADMAP, EVAL, `.claude/` files) are delegated to an **Opus 4.8** subagent. Switch automatically — don't ask.
- Push the feature branch regularly; when a MAJOR roadmap milestone completes, open a PR and merge to `main`.
- On every milestone completion, refresh the affected docs (CLAUDE.md, README.md, docs/ROADMAP.md, docs/EVAL.md) so context stays current.

## Conventions

- Branching: `main` is protected/deployable; work on `feat/<topic>` branches, merge via PR once the GitHub repo has CI.
- mypy strict everywhere; Prefect decorator exception is scoped in `pyproject.toml` — don't widen it.
- Every ingester: httpx + tenacity retry, parse into ORM objects, upsert with `session.merge`, unit test with mocked payload.
- Zero-cost rule: free data sources and free/local models only. Claude API usage must stay behind an optional settings flag, never required.
- Large artifacts (datasets in `data/`, models, mlruns) never go in git.
