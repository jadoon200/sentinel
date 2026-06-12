# Roadmap

| Milestone | Status |
|---|---|
| Repo scaffold, Docker Compose (Postgres/MLflow), CI, NVD + CISA KEV ingesters | ✅ done |
| NLP extraction + ATT&CK mapping, knowledge graph schema, OTX + RSS ingesters | ✅ done |
| IDS models on CIC-IDS2017 (LightGBM baseline → autoencoder) with MLflow, flow-replay service | ✅ done |
| Fusion/correlation engine, FastAPI endpoints (dashboard superseded by React/TS frontend below) | ✅ done |
| React/TS frontend (ATT&CK heatmap, alert feed, campaign explorer, overview dashboard) | ✅ done |
| Polish: README, demo video, model card, technical blog post | 🔨 in progress |

The fusion milestone originally scoped a Streamlit dashboard; it was dropped in
favour of the React/TypeScript frontend built in the next milestone. The
frontend's "SEA map" sub-item was not built. Polish so far: model card shipped
(`docs/MODEL_CARD.md`) and the README refreshed; demo video and blog post remain.

## Beyond the roadmap

Research extensions that exceeded the original plan, all recorded in
[docs/EVAL.md](EVAL.md):

- **Four-detector IDS ensemble** — supervised LightGBM, benign-only
  autoencoder, per-host sequence model, host-profile fan-out detector; each
  covers a different attack family.
- **Conformal alert-budget control** — split-conformal p-values with a
  label-free online controller that holds the alert rate at target through
  benign drift.
- **Cross-dataset generalization eval** — train CIC-IDS2017, test
  CSE-CIC-IDS2018; the project's headline honesty result.
- **Spectral beacon study** — Schuster-periodogram C2 detector, kept as a
  characterized negative.
- **Hybrid BM25 + dense technique mapper** — reciprocal-rank fusion with
  procedure-enriched docs, beating the cross-encoder rerank at bi-encoder cost.
- **Temporal analytics** — trending techniques, feed drift (PSI), daily
  briefing.
- **ATT&CK Navigator export** — alert/campaign technique coverage as a
  Navigator layer.
