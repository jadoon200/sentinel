# SENTINEL

**Cyber threat intelligence fusion platform** — correlates open-source threat intelligence (OSINT) with ML-based network intrusion detection, the way real SOCs and intelligence fusion centres do.

> 🚧 OSINT ingestion + NLP technique mapping (SecureBERT 2.0 retrieval + reranking over the full ATT&CK catalog) + campaign correlation are done; ML intrusion-detection models (CIC-IDS2017) are next. See [docs/ROADMAP.md](docs/ROADMAP.md).

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

Python 3.12 (conda) · SQLAlchemy/Alembic · PostgreSQL · Prefect · httpx · MLflow · LightGBM/PyTorch · FastAPI · React/TypeScript · Docker Compose · GitHub Actions

All data sources are free: [NVD CVE API](https://nvd.nist.gov/developers/vulnerabilities), [CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog), [MITRE ATT&CK](https://attack.mitre.org/), [AlienVault OTX](https://otx.alienvault.com/), [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html).

## Quickstart

```bash
# 1. create + activate the sentinel env, install deps
make env
conda activate sentinel
make install

# 2. start Postgres (+ MLflow) and run migrations
make up

# 3. run the OSINT ingestion flow (no API keys required)
make ingest

# 4. NLP technique tagging + campaign correlation over ingested reports
make enrich

# 5. checks
make check
```

Configuration via `.env` — see [.env.example](.env.example).

The technique mapper is zero-shot (no training data); benchmarked against [TRAM](https://github.com/center-for-threat-informed-defense/tram), parent-level hit@5 = 0.58 retrieval-only / 0.66 with reranking — see [docs/EVAL.md](docs/EVAL.md).
