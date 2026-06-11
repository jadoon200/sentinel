# SENTINEL

**Cyber threat intelligence fusion platform** — correlates open-source threat intelligence (OSINT) with ML-based network intrusion detection, the way real SOCs and intelligence fusion centres do.

> 🚧 OSINT ingestion + NLP technique mapping (SecureBERT 2.0 retrieval + reranking over the full ATT&CK catalog) + campaign correlation are done, as are the ML intrusion-detection models (CIC-IDS2017) with an honest temporal-split evaluation; the fusion layer (read-only API done, dashboard next) is in progress. See [docs/ROADMAP.md](docs/ROADMAP.md).

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

Python 3.12 (conda) · SQLAlchemy/Alembic · PostgreSQL · Prefect · httpx · MLflow · LightGBM/PyTorch · MLX (optional, Apple silicon) · FastAPI · React/TypeScript · Docker Compose · GitHub Actions

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

# 5. IDS models on CIC-IDS2017 — download the corrected dataset zip from
#    intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip
#    into data/cicids2017/, then:
make train          # LightGBM binary baseline (--split temporal for the unseen-family eval)
make train-anomaly  # benign-only autoencoder anomaly detector
make replay         # persist top detections as ATT&CK-tagged alerts
make api            # serve the read-only knowledge-graph API on :8000

# 6. checks
make check
```

Configuration via `.env` — see [.env.example](.env.example).

The technique mapper is zero-shot (no training data); benchmarked against [TRAM](https://github.com/center-for-threat-informed-defense/tram), parent-level hit@5 = 0.58 retrieval-only / 0.66 with reranking — see [docs/EVAL.md](docs/EVAL.md).

The IDS baseline scores 0.9998 ROC-AUC on a random split but collapses to F1 0.001 under a temporal split (train Mon–Wed, test Thu–Fri unseen attack families) — the within-dataset inflation reproduced deliberately; the benign-only autoencoder recovers Infiltration 0.84 / DDoS 0.71 / XSS 0.67 recall on those unseen families — see [docs/EVAL.md](docs/EVAL.md).

The autoencoder runs on MLX or torch-MPS; MLX is the auto-selected default on Apple silicon after a 5-seed benchmark (recall parity, 3.7× faster, no OpenMP clash with LightGBM) — see [docs/EVAL.md](docs/EVAL.md).
