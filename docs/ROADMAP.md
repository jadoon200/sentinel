# Roadmap

| Milestone | Status |
|---|---|
| Repo scaffold, Docker Compose (Postgres/MLflow), CI, NVD + CISA KEV ingesters | ✅ done |
| NLP extraction + ATT&CK mapping, knowledge graph schema, OTX + RSS ingesters | ✅ done |
| IDS models on CIC-IDS2017 (LightGBM baseline → autoencoder) with MLflow, flow-replay service | ✅ done |
| Fusion/correlation engine, FastAPI endpoints (dashboard superseded by React/TS frontend below) | ✅ done |
| React/TS frontend — question-led three-tab storyline (Threat feed / Landscape / Model report card) | ✅ done |
| Polish: README, demo video, model card, technical blog post | 🔨 in progress |

The fusion milestone originally scoped a Streamlit dashboard; it was dropped in
favour of the React/TypeScript frontend built in the next milestone. The
frontend's "SEA map" sub-item was not built. Polish so far: model card shipped
(`docs/MODEL_CARD.md`) and the README refreshed; demo video and blog post remain.

## Beyond the roadmap

Research extensions that exceeded the original plan, all recorded in
[docs/EVAL.md](EVAL.md):

- **Five-detector IDS ensemble** — supervised LightGBM, benign-only
  autoencoder, per-host sequence model, host-profile fan-out detector, and the
  beacon-dispersion detector; each covers a different attack family, and together
  they cover 7/7 unseen Thu–Fri families at recall ≥0.93 (`make eval-ensemble`).
- **Conformal alert-budget control** — split-conformal p-values with a
  label-free online controller that holds the alert rate at target through
  benign drift.
- **Cross-dataset generalization eval** — train CIC-IDS2017, test
  CSE-CIC-IDS2018; the project's headline honesty result.
- **Cross-network transfer fix** — the headline. A 2017-trained IDS detects
  nothing on a different network at any usable threshold; every label-free fix
  (CORAL, transfer-stable features, target-trained autoencoder) failed, but
  ~50 labelled target flows recover 0.95–0.99 recall across three attack
  families (brute-force, DoS, Bot) on contamination-free held-out splits.
  Cross-network IDS transfer is a few-shot labelling problem
  (`sentinel/ids/domain_adapt.py`).
- **Deep fusion scoring** — the headline join, hardened past set overlap. Each
  alert↔campaign match is scored by a calibrated fusion strength = technique
  rarity (IDF over the report corpus) × campaign recency (age decay) ×
  corroboration, combined as a geometric mean with every component exposed for
  explainability (`sentinel/correlate/fusion.py`). Answers the reviewer's
  "lots of campaigns involve DoS — why is this match meaningful?" with a number.
- **Host-fusion threat rollups** — the dashboard's fusion unit: per-flow alerts
  roll up into per-host threats joined to CTI campaigns, with detector
  agreement, unioned ATT&CK techniques, and a transparent risk score scaled by
  the best campaign's fusion strength.
- **SQL injection — flow ceiling → payload detector.** SQLi is invisible to
  every flow detector (12 flows, none in training, indistinguishable from benign
  HTTP — a feature ceiling), so it gets a different modality: a char-n-gram +
  logistic-regression payload (WAF) detector (`ids/sqli.py`, T1190) validated
  cross-corpus at F1 0.984/0.998 on two free public payload sources.
- **Beacon detection — periodicity negative → dispersion fix.** Three
  periodicity attempts (variance pairs, Schuster periodogram) only *ranked* Bot
  (AUC ≤0.83, recall ≤0.056) — benign timers out-periodic the beacon. Re-framing
  to per-channel **data-size dispersion** (`ids/beacon.py`) lifts Bot channel
  recall to 5/5 @~1.6% FPR (AUC 0.995): an ARES C2 channel mixes empty polls and
  data tasking, so its byte-size CV is extreme. Foothold (only 5 C2 channels);
  mechanism confirmed on 2018 Bot.
- **Hybrid BM25 + dense technique mapper** — reciprocal-rank fusion with
  procedure-enriched docs, beating the cross-encoder rerank at bi-encoder cost.
- **Temporal analytics** — trending techniques, feed drift (PSI, additively
  smoothed), daily briefing; windowed on publish date so a one-shot ingest
  still yields a real timeline.
- **Multi-source CTI ingestion** — 28 keyless RSS/Atom feeds (vendor research +
  CERTs) with per-publisher provenance, browser-UA fetching, and a cron-friendly
  `make refresh` (ingest → enrich → replay). A refresh pulls ~600 reports across
  ~29 sources; OTX adds label-bearing pulses once a free key is set.
- **ATT&CK Navigator export** — alert/campaign technique coverage as a
  Navigator layer.
