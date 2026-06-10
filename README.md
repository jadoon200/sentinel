# SENTINEL

**Cyber threat intelligence fusion platform** — correlates open-source threat intelligence (OSINT) with ML-based network intrusion detection, the way real SOCs and intelligence fusion centres do.

> 🚧 Week 1 of 8 — OSINT ingestion pipeline (NVD + CISA KEV). See [docs/ROADMAP.md](docs/ROADMAP.md).

## Architecture

```
 ┌─────────────────────────┐      ┌──────────────────────────┐
 │  Layer 1 · OSINT intel  │      │ Layer 2 · Intrusion det. │
 │  NVD / CISA KEV / OTX / │      │ CIC-IDS flows → LightGBM │
 │  ATT&CK / RSS → NLP →   │      │ + autoencoder → alerts   │
 │  threat knowledge graph │      │ tagged w/ ATT&CK techn.  │
 └───────────┬─────────────┘      └────────────┬─────────────┘
             └────────────┬───────────────────┘
                          ▼
             ┌─────────────────────────┐
             │  Layer 3 · Fusion       │
             │  correlation engine +   │
             │  dashboard & briefings  │
             └─────────────────────────┘
```

## Stack

Python 3.12 · uv · SQLAlchemy/Alembic · PostgreSQL · Prefect · httpx · MLflow · LightGBM/PyTorch · FastAPI · React/TypeScript · Docker Compose · GitHub Actions

All data sources are free: [NVD CVE API](https://nvd.nist.gov/developers/vulnerabilities), [CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog), [MITRE ATT&CK](https://attack.mitre.org/), [AlienVault OTX](https://otx.alienvault.com/), [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html).

## Quickstart

```bash
# 1. install deps
uv sync

# 2. start Postgres (+ MLflow) and run migrations
make up

# 3. run the OSINT ingestion flow (no API keys required)
make ingest

# 4. checks
make check
```

Configuration via `.env` — see [.env.example](.env.example).
