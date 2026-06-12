# SENTINEL

**Cyber threat intelligence fusion platform** — correlates open-source threat intelligence (OSINT) with ML-based network intrusion detection, the way real SOCs and intelligence fusion centres do.

> OSINT ingestion, NLP technique mapping over the full ATT&CK catalog, campaign correlation, a four-detector IDS ensemble with an honest cross-dataset/temporal evaluation, conformal alert-budget control, temporal analytics, a read-only knowledge-graph API, and a React/TypeScript dashboard are all in place. Remaining polish: demo video + blog post. See [docs/ROADMAP.md](docs/ROADMAP.md).

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

## What makes this different

- **Honest-evaluation discipline.** Every model is reported on the number that
  survives a network change, not a headline AUC: cross-dataset (2017 → 2018)
  transfer, temporal splits over unseen attack families, multi-seed backend
  benchmarks, and recorded *negative* results (spectral beacons, botnet recall).
  The full record is [docs/EVAL.md](docs/EVAL.md) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md).
- **One ATT&CK graph.** OSINT × NLP × IDS are fused into a single
  technique-keyed knowledge graph — campaign correlation, alert context, and
  KEV-weighted briefings all join on the same ATT&CK technique IDs.

## Headline results

All numbers from [docs/EVAL.md](docs/EVAL.md), stated honestly.

- **Cross-dataset 2017 → 2018 (the centrepiece honesty test).** Within-2017
  brute-force is perfectly separable (ROC-AUC **1.0000**); the model still
  *ranks* 2018 attacks above 2018 benign (cross-AUC **0.940**) but at the
  deployed operating point detects none of them (**recall @ 1% FPR 0.000**) —
  the absolute threshold learned on 2017 lands in the wrong place for 2018's
  score distribution. This is the published within-dataset-inflation failure
  reproduced first-hand, and it motivates the conformal controller below.
- **Conformal alert-budget control (the fix).** A label-free online controller
  re-derives the operating point from the target network's own benign traffic,
  holding the alert rate at a 1% budget through the same drift (FPR 1.10%) while
  rare attacks keep alerting (Infiltration 0.84, XSS 0.70).
- **IDS temporal-split honesty.** A LightGBM baseline scores ROC-AUC up to
  1.0000 within-dataset but its *default* threshold collapses to F1 0.001 on
  unseen Thu–Fri attack families — the same calibration story. Re-calibrating
  the threshold from benign traffic alone (no attack labels) recovers F1 0.800.
- **Four-detector ensemble, per-family coverage.** Detectors cover different
  families by construction: supervised LightGBM (seen families ≈ 1.0),
  benign-only autoencoder (Infiltration 0.84 / DDoS 0.71 / XSS 0.67), per-host
  sequence model (XSS 1.00 / Web Brute Force 0.94), host-profile fan-out
  detector (PortScan 0.998).
- **Technique mapper, hybrid retrieval.** Zero-shot mapping over the full
  enterprise ATT&CK catalog (697 techniques), benchmarked on 10,411 TRAM
  sentences: BM25 + dense reciprocal-rank fusion with procedure-enriched docs
  reaches parent-level hit@5 **0.690** — beating a 130× more expensive
  cross-encoder rerank at bi-encoder cost.
- **MLX vs torch backend adoption.** The autoencoder's MLX port is the
  auto-selected default on Apple silicon after a 10-seed benchmark: recall
  parity, 3.3× faster training, and no OpenMP clash with LightGBM (torch remains
  the Linux/CI fallback).

## Stack

Python 3.12 (conda) · SQLAlchemy/Alembic · PostgreSQL · Prefect · httpx · MLflow · LightGBM/PyTorch · MLX (optional, Apple silicon) · FastAPI · React/TypeScript · Docker Compose · GitHub Actions

All data sources are free: [NVD CVE API](https://nvd.nist.gov/developers/vulnerabilities), [CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog), [MITRE ATT&CK](https://attack.mitre.org/), [AlienVault OTX](https://otx.alienvault.com/), RSS/Atom CTI feeds, [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html), and [TRAM](https://github.com/center-for-threat-informed-defense/tram).

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

### IDS models

Download the corrected CIC-IDS2017 dataset zip from
`intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip` into
`data/cicids2017/`, then:

```bash
make train           # LightGBM binary baseline (--split temporal for the unseen-family eval)
make train-anomaly   # benign-only autoencoder anomaly detector (MLX / torch-MPS)
make train-sequence  # per-host sequence model (MLX gated recurrence)
make train-profile   # host-profile fan-out detector (PortScan)
make replay          # persist top detections as ATT&CK-tagged alerts
make eval-cross      # cross-dataset 2017 → 2018 generalization (downloads a 2018 day)
```

### API + dashboard

```bash
make api        # read-only knowledge-graph API on :8000 (needs make up)
make ui         # React dashboard dev server on :5173 (needs make api running)
make briefing   # print the auto-generated daily threat briefing
```

API endpoints: `/stats`, `/campaigns` (+ `/{id}`), `/reports`, `/alerts`
(+ `/{id}/context` for technique fusion), `/techniques` (+ `/{id}`),
`/trending`, `/feed-drift`, `/briefing`, and `/attack-navigator-layer`
(ATT&CK Navigator export of alert/campaign technique coverage).

The dashboard surfaces an ATT&CK heatmap, alert feed, campaign explorer, and an
overview dashboard over those endpoints.
