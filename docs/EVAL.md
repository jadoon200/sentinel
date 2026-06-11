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
