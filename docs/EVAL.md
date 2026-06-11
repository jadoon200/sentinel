# Model evaluations

## IDS baseline — LightGBM on corrected CIC-IDS2017

Trained on the corrected dataset (Engelen et al., WTMC 2021 — >20% of original
flows relabeled/fixed), "Attempted" flows dropped, identifier/topology columns
(IPs, ports, timestamps) excluded to prevent testbed shortcut learning.
Reproduce: `python -m sentinel.ids.train [--split temporal]`. Tracked in
MLflow (`ids-lightgbm-baseline`).

| Run | ROC-AUC | PR-AUC | F1 @0.5 | FPR |
|---|---|---|---|---|
| Random 80/20 split (full 2.2M flows) | 0.9998 | 0.9994 | 0.988 | 0.4% |
| Temporal split (train Mon–Wed, test Thu–Fri) | 0.9895 | 0.972 | **0.001** | 0.00% |

The random split looks near-perfect but per-class recall already shows cracks
(XSS 0.20, Infiltration 0.88). The temporal split is the honest number: every
Thu/Fri attack family (web attacks, infiltration, botnet, portscan, DDoS) is
absent from training, and at the deployment threshold the classifier detects
**none of them** — scores rank attacks above benign (ROC-AUC 0.99) but far
below the threshold calibrated on seen attacks. This is exactly the
within-dataset-inflation failure documented for NIDS literature, reproduced
here on purpose: it motivates the anomaly-detection track and threshold
calibration, and it's why SENTINEL reports temporal/cross-dataset numbers
instead of headline AUCs.

# Technique mapper evaluation

Zero-shot mapping of CTI sentences to ATT&CK techniques, evaluated against
[TRAM](https://github.com/center-for-threat-informed-defense/tram) bootstrap
data (11,130 sentences hand-labeled by analysts, 50 techniques). The mapper
retrieves against the **full enterprise catalog (697 active techniques)** with
no task-specific training — unlike classifier approaches (TRAM itself,
TTPxHunter) that are trained on and restricted to the ~50 most common
techniques.

Reproduce: `python scripts/eval_mapper.py --sample 2000` (seed 13).

## Bi-encoder retrieval (cisco-ai/SecureBERT2.0-biencoder), 2,000 sentences

hit@k = a gold technique appears in the top-k predictions; parent = credit at
parent-technique level (T1059.001 → T1059).

| k | hit@k | parent hit@k |
|---|---|---|
| 1 | 0.224 | 0.342 |
| 3 | 0.358 | 0.502 |
| 5 | 0.435 | 0.580 |
| 10 | 0.538 | 0.679 |

~37 ms/sentence on Apple Silicon CPU after a one-off ~50 s catalog embedding.

## + cross-encoder reranking (SecureBERT2.0-cross_encoder), 300 sentences

Top-20 retrieval candidates reranked pairwise (`--sample 300 --rerank`,
same seed/shuffle — smaller sample, so noisier).

| k | hit@k | parent hit@k |
|---|---|---|
| 1 | 0.230 | 0.393 |
| 3 | 0.400 | 0.587 |
| 5 | 0.493 | 0.663 |
| 10 | 0.563 | 0.743 |

Reranking buys ~+8 pp parent hit@3–5 over retrieval alone, at ~1.8 s/sentence
on CPU (~50× the bi-encoder cost) — right default for report-level ingestion,
optional for interactive use.

Notes:

- Single-sentence scores are the floor: SENTINEL corroborates evidence across
  the sentences/reports of a campaign (`aggregate_matches`), which is worth
  roughly +26% F1 in multi-report settings (arXiv:2604.07470).
- TRAM sentences are short and often context-free ("network traffic
  communicates over a raw socket"), so exact sub-technique hit@1 understates
  document-level usefulness; parent-level hit@5 (0.58) is the better proxy for
  the triage use case (suggest candidate techniques to an analyst).
