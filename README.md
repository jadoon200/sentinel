# SENTINEL

**Cyber threat intelligence fusion platform** — correlates open-source threat intelligence (OSINT) with ML-based network intrusion detection, the way real SOCs and intelligence fusion centres do.

> OSINT ingestion, NLP technique mapping over the full ATT&CK catalog, campaign correlation, a five-detector IDS ensemble with an honest cross-dataset/temporal evaluation, a measured cross-network transfer fix (few-shot domain adaptation), conformal alert-budget control, host-fusion threat rollups, temporal analytics, a read-only knowledge-graph API, and a React/TypeScript dashboard are all in place. Remaining polish: demo video + blog post. See [docs/ROADMAP.md](docs/ROADMAP.md).

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
- **One ATT&CK graph, scored fusion.** OSINT × NLP × IDS are fused into a single
  technique-keyed knowledge graph — campaign correlation, alert context, and
  KEV-weighted briefings all join on the same ATT&CK technique IDs. The join is
  not raw tag overlap: each alert↔campaign match carries a calibrated fusion
  strength = technique rarity (IDF) × campaign recency × corroboration, so a
  specific, active correlation outranks a coincidental shared tag.

## Headline results

All numbers from [docs/EVAL.md](docs/EVAL.md), stated honestly.

- **Cross-network transfer, failure → fix (the centrepiece research result).**
  A 2017-trained IDS is perfectly separable within-dataset (ROC-AUC **1.0000**)
  but at any usable threshold detects *none* of the same attacks on a different
  network in 2018 (**recall @ 1% FPR 0.000**) — the absolute threshold lands in
  the wrong place for 2018's score distribution. Every **label-free** fix failed:
  CORAL covariance alignment, transfer-stable feature selection, and a
  target-trained autoencoder all stayed at recall ~0. **Few-shot is the fix** —
  50 labelled target flows recover **0.95–0.99 recall across three different
  attack families** (brute-force, DoS, Bot) on contamination-free held-out
  splits; Bot's blind-2017 baseline ranks *worse than chance* (AUC 0.40) and 50
  labels lift it to AUC 0.997. Cross-network IDS transfer is a few-shot
  *labelling* problem, not a representation-alignment one. The labelling budget
  is small and measured (`make eval-label-efficiency`, 5 seeds): **~50 labels
  reach ≥0.88 recall, ~100 reach ≥0.97**, and *active* (uncertainty) selection
  underperforms random — a transfer-collapsed model's confidence can't pick
  informative flows, so random balanced sampling wins.
- **Conformal alert-budget control (within-network).** A label-free online
  controller re-derives the operating point from the target network's own benign
  traffic, holding the alert rate at a 1% budget through within-network drift
  (FPR 1.10%) while rare attacks keep alerting (Infiltration 0.84, XSS 0.70) —
  the answer to drift *within* a network, measured to its limit against the
  cross-network case above.
- **Host-fusion threat rollups.** Per-flow alerts roll up into per-host threats:
  each host shows which detectors agree, its unioned ATT&CK techniques, a
  transparent risk score, and the real-world CTI campaign it fuses with — each
  campaign link scored by a calibrated fusion strength (rarity × recency ×
  corroboration), so the rollup ranks meaningful correlations, not keyword
  collisions. Worked example and table in [docs/EVAL.md](docs/EVAL.md).
- **IDS temporal-split honesty.** A LightGBM baseline scores ROC-AUC up to
  1.0000 within-dataset but its *default* threshold collapses to F1 0.001 on
  unseen Thu–Fri attack families — the same calibration story. Re-calibrating
  the threshold from benign traffic alone (no attack labels) recovers F1 0.800.
- **Five-detector ensemble, per-family coverage.** Detectors cover different
  families by construction: supervised LightGBM (seen families ≈ 1.0),
  benign-only autoencoder (Infiltration 0.84 / DDoS 0.71 / XSS 0.67), per-host
  sequence model (XSS 1.00 / Web Brute Force 0.94), host-profile fan-out
  detector (PortScan 0.998), and a data-size-dispersion beacon detector that
  lifts CIC Bot channel recall from ~0 to 5/5 @1.6% FPR. Cross-validated on
  CTU-13 (7 botnet families, 1,470 channels) it **does not generalize** (0.010 —
  the signature is ARES-specific): a measured limitation, not an assumed one.
- **SQL injection, by its payload signature.** SQLi is invisible to the
  *unsupervised* flow detectors (12 flows, none in training, benign-looking on
  volume/timing) — a calibrated supervised model flags the 12 but only on
  within-dataset flows. Robust SQLi detection gets a payload (WAF-style) detector:
  char n-grams + logistic regression over request strings, mapped to T1190 and
  validated **cross-corpus** (train one public payload source, test another) at
  F1 **0.984 / 0.998** — generalization, not memorization. Wired into the platform
  via a WAF replay (`make waf-replay`) → T1190 alerts that fuse with campaigns and
  show in the threat feed.
- **Ensemble coverage, not single-model recall.** No single detector covers the
  unseen attack families (the best unsupervised model averages 0.268), but the
  five-detector ensemble covers **7/7 unseen families at recall ≥ 0.93**, each by
  its specialist (`make eval-ensemble`) — the system catches what no one model can.
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

All data sources are free: [NVD CVE API](https://nvd.nist.gov/developers/vulnerabilities), [CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog), [MITRE ATT&CK](https://attack.mitre.org/), **28 keyless CTI RSS/Atom feeds** (vendor research blogs + CERTs — Talos, Unit42, Mandiant, CrowdStrike, Securelist, Project Zero, NCSC-UK, …), [AlienVault OTX](https://otx.alienvault.com/) (optional free key), [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html), public SQLi payload corpora (HttpParamsDataset, Kaggle SQLiV2), and [TRAM](https://github.com/center-for-threat-informed-defense/tram). A typical refresh ingests ~600 reports across ~29 sources into the graph; `make refresh` runs the full ingest → enrich → replay pipeline (cron-friendly).

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
make eval-cross         # cross-dataset 2017 → 2018 generalization (downloads a 2018 day)
make eval-domain        # label-free domain-adaptation fixes vs few-shot (2017 → 2018)
make eval-cross-family  # cross-family stress test: few-shot across brute-force / DoS / Bot
```

### API + dashboard

```bash
make api        # read-only knowledge-graph API on :8000 (needs make up)
make ui         # React dashboard dev server on :5173 (needs make api running)
make briefing   # print the auto-generated daily threat briefing
```

API endpoints: `/health`, `/stats`, `/campaigns` (+ `/{id}`), `/reports`,
`/alerts` (+ `/{id}/context` for scored technique fusion), `/hosts` and
`/hosts/simulated` (host-fusion threat rollups), `/techniques` (+ `/{id}`),
`/trending`, `/feed-drift`, `/briefing`, and `/attack-navigator-layer`
(ATT&CK Navigator export of alert/campaign technique coverage).

The dashboard is a question-led three-tab storyline over those endpoints:

- **Threat feed** — the fusion view. Per-host threat rollups: each host shows
  which of the five detectors agree, its unioned ATT&CK techniques, a
  transparent risk score, and the CTI campaign it fuses with; expandable into a
  left-to-right evidence chain (detectors → host + techniques → matched
  real-world campaign, with a fusion-strength meter and its rarity/recency
  breakdown on each match), with a "simulate detection" button that reveals
  held-out detections.
- **Landscape** — trending techniques, feed drift (PSI), the daily briefing, and
  ATT&CK Navigator export.
- **Model report card** — the honest evaluation story, including the
  cross-network failure *and* its few-shot fix, plus a **"Try the mapper"** panel:
  paste any CTI paragraph and the live zero-shot mapper ranks the closest ATT&CK
  techniques (`POST /map-techniques`). It inspects only the pasted text — it does
  not fetch or scan a URL — so the API stays effectively read-only.

### Deploying the API publicly

The inference route runs a model, so the API ships with graceful-degradation
guards — configurable CORS origins, a request-size cap, per-client rate limiting,
and a bounded-concurrency cap that sheds load as `503` rather than exhausting
memory — all local-safe by default and tuned via `SENTINEL_API_*` env vars. See
[docs/DEPLOY.md](docs/DEPLOY.md) for the env vars and the reverse-proxy / TLS /
sizing steps to do at deploy time.
