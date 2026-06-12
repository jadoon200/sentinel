# SENTINEL — Model Card

This card documents the five models in the SENTINEL threat-intelligence fusion
platform. Every number is traceable to [`docs/EVAL.md`](EVAL.md), the complete
evaluation record; this card summarizes, it does not introduce new results.

## Intended use & users

- **Use case:** SOC triage support and threat-intelligence fusion — surfacing
  candidate intrusions, ranking them, and tagging them with ATT&CK techniques
  so an analyst can correlate detections against ingested OSINT campaigns.
- **Users:** security analysts (decision-support), and researchers/educators
  studying NIDS evaluation pitfalls and zero-shot CTI mapping.
- **Out of scope:** these models are **not** autonomous blockers, firewalls, or
  ground-truth labelers. Outputs are ranked suggestions requiring human review.
  None should gate enforcement actions without analyst confirmation and
  target-network re-calibration (see Caveats).

## Model details

### 1. ATT&CK technique mapper (zero-shot)

- **Task:** map a CTI sentence to MITRE ATT&CK techniques over the **full
  enterprise catalog (697 active techniques)** — not the ~50-technique subset
  that trained classifiers (TRAM, TTPxHunter) are restricted to.
- **Architecture:** SecureBERT 2.0 bi-encoder dense retrieval, fused with BM25
  lexical ranks via reciprocal-rank fusion (RRF). Technique documents are
  enriched with real ATT&CK procedure examples; the lexical side exploits the
  procedure vocabulary (tool names, commands, registry paths). Optional
  cross-encoder pairwise rerank of the top-20 candidates. Code:
  `src/sentinel/nlp/{mapper,encoders,lexical}.py`.
- **Training data:** none — zero-shot retrieval, no task-specific fine-tuning.
- **Eval protocol:** TRAM bootstrap annotations (11,130 analyst-labeled
  sentences; 10,411 after a ≥4-word filter), seed 13. hit@k = a gold technique
  in top-k; parent = credit at parent-technique level.
- **Key numbers (adopted config = procedures + hybrid):** hit@5 **0.553** /
  parent@5 **0.690**; hit@10 0.682 / parent@10 0.792. This beats the 130×
  more expensive cross-encoder rerank (0.543 / 0.663 at @5) at bi-encoder cost.
  Retrieval ~12 ms/sentence; rerank ~1.6 s/sentence (Apple silicon).
- **Design note:** the mapper ranks by RRF but reports the **dense cosine** as
  confidence, because RRF scores are rank-based and carry no absolute scale —
  downstream tagging thresholds stay on the cosine scale.

### 2. Supervised IDS baseline (LightGBM)

- **Task:** binary benign-vs-attack flow classification.
- **Architecture:** LightGBM gradient-boosted trees. Code:
  `src/sentinel/ids/train.py`.
- **Training data:** corrected CIC-IDS2017 (Engelen et al., WTMC 2021 —
  >20% of original flows relabeled). "Attempted" flows dropped;
  identifier/topology columns (IPs, ports, timestamps) excluded to prevent
  testbed shortcut learning.
- **Eval protocol:** random 80/20 split vs **temporal split** (train Mon–Wed,
  test Thu–Fri — every test attack family unseen in training). MLflow:
  `ids-lightgbm-baseline`.
- **Key numbers:** random split ROC-AUC 0.9998, F1 0.988 @0.5, FPR 0.4%.
  Temporal split: ROC-AUC 0.9895 but F1 **0.001** @0.5 (the deliberate
  within-dataset-inflation collapse). With a benign-calibrated threshold
  (`--calibrate-fpr 0.01`), F1 recovers to **0.800** at 1.5% FPR — ranking
  transfers to unseen families far better than the default threshold suggests;
  the collapse is mostly threshold miscalibration, fixable from benign traffic
  alone.

### 3. Benign-only autoencoder (anomaly detector)

- **Task:** flag flows whose reconstruction error exceeds the 99th percentile
  of held-out benign error — detecting **unseen** attack families.
- **Architecture:** MLP autoencoder, two interchangeable backends — MLX
  (`src/sentinel/ids/anomaly_mlx.py`, default on Apple silicon) and torch-MPS
  (`anomaly.py`, fallback / Linux CI). Identical architecture and protocol.
- **Training data:** Mon–Wed **benign flows only** (no attack labels).
- **Eval protocol:** identical temporal split as the supervised baseline;
  per-family recall on Thu–Fri. Backend choice gated on a 10-seed benchmark
  (`scripts/bench_anomaly.py`). MLflow: `ids-autoencoder`.
- **Key numbers (per-family recall @p99):** Infiltration **0.844**, DDoS
  **0.705**, XSS 0.667, Web Brute Force 0.477, SQLi 0.000, Bot 0.060,
  PortScan 0.007. Overall recall 0.268 at 6.3% FPR.
- **Backend adoption:** MLX vs torch-MPS at 10 seeds — ROC-AUC parity
  (0.913 vs 0.915), recall parity (MLX nominally ahead, 0.275 vs 0.245),
  **3.3× faster** training (1.18 s vs 3.93 s) with lower variance. MLX links
  no OpenMP, so it shares a process with LightGBM; torch deadlocks there
  (duplicate libomp on macOS), which hung the replay service before the switch.

### 4. Sequence model (per-host gated recurrence)

- **Task:** score attacks that are anomalous **as a sequence from one host**
  even when individual flows look benign.
- **Architecture:** input-gated recurrent cell (MLX) predicting each next flow
  from the host's preceding window; window score = next-step prediction error,
  two-sided, with a leakage-free per-host **inter-arrival Δt** feature
  (relative, not absolute). Default `--window 32 --stride 16`. Code:
  `src/sentinel/ids/sequence.py`.
- **Training data:** Mon–Wed benign flows only, grouped per host.
- **Eval protocol:** temporal split, FPR ≈ 2%, window labeled by its last
  flow; seed stability checked (5 seeds at w16, 3 at w32).
- **Key numbers (+ inter-arrival config):** XSS **1.000** (1.000 in every
  run), Web Brute Force 0.824 (0.96 ± 0.06 at w32), DDoS 0.486 (inter-arrival
  raised it from 0.021 — flood cadence is a timing signature), Infiltration
  0.333. These are genuine unique adds over the autoencoder for web attacks.
- **Honest negative:** PortScan / Bot / SQLi stay at **0.000** across all
  variants — scan/beacon windows are *more predictable than benign traffic*
  (ROC-AUC ≈ 0.42, inverted); prediction-error magnitude cannot represent
  "suspiciously machine-like".

### 5. Host-profile fan-out detector (explainable statistics)

- **Task:** detect the scan/beacon families the neural detectors miss, with no
  neural net.
- **Architecture:** four explainable per-window statistics over each host's
  stream (unique dst ports, unique dst IPs, log flow rate, log mean packet
  count), robust-scaled on benign Mon–Wed windows; alert on excess max
  robust-z. Destinations feed *counts* only — never raw feature values. Code:
  `src/sentinel/ids/profile.py`.
- **Training data:** benign Mon–Wed windows (scaling calibration only).
- **Eval protocol:** temporal split; per-host grouping (deployed default) and
  per-(src→dst) pair grouping for the beacon follow-up.
- **Key numbers:** PortScan recall **0.998** (0.000 for every other detector).
  Nominal FPR 12.1%, but **1.15%** with host 192.168.10.8 excluded.
- **Ground-truth label-gap discovery:** ~90% of the "false positives" trace to
  one host, 192.168.10.8, on Thursday — the documented **Infiltration victim**,
  which port-scans the internal subnet after the meterpreter compromise. Those
  flows are labeled BENIGN in the corrected ground truth; the detector is
  flagging real lateral scanning the labels miss.
- **Beacon foothold:** per-(src→dst) pair grouping puts Bot recall above zero
  for the first time (**0.056** at 0.9% FPR, window ROC-AUC 0.83) by isolating
  each channel; per-host windows interleave benign traffic and destroy the
  timer pattern. Ranking signal exists, operating-point recall does not yet —
  host grouping stays the deployed default.

## Evaluation summary

| Model | Protocol | Headline metric | Honest caveat |
|---|---|---|---|
| ATT&CK mapper | TRAM, 10,411 sentences, zero-shot | hit@5 0.553 / parent@5 0.690 | sub-technique hit@1 0.277 understates doc-level use |
| LightGBM IDS | CIC-IDS2017 temporal split | F1 0.800 @1.5% FPR (calibrated) | F1 0.001 at default threshold |
| LightGBM IDS | CIC-IDS2017 random split | ROC-AUC 0.9998, F1 0.988 | inflated; XSS recall already 0.20 |
| Autoencoder | temporal, unseen families | Infiltration 0.844 / DDoS 0.705 | overall 0.268 recall @6.3% FPR |
| Sequence model | temporal, per-host windows | XSS 1.000 / Brute Force ~0.96 | scan/beacon 0.000 (inverted) |
| Host-profile | temporal, fan-out stats | PortScan 0.998 @1.15% FPR | Bot 0.000 deployed (0.056 per-pair) |

## Known limitations & failure modes

- **Bot/beacon recall ≈ 0** at deployed operating points across all detectors;
  only the extreme tail of per-pair periodicity scoring clears the threshold
  (the ARES beacon's cadence overlaps benign periodic services like NTP).
- **SQL Injection recall = 0** for both the autoencoder and the sequence model.
- **FPR drift under distribution shift:** the autoencoder's observed 6.3% FPR
  exceeds the calibrated 1% because Thu–Fri benign traffic differs from Mon–Wed
  benign traffic. Thresholds calibrated on one period do not transfer cleanly.
- **Single-testbed dataset:** all IDS results are CIC-IDS2017 only; the
  documented within-dataset inflation means random-split AUCs do not predict
  cross-network performance.
- **Imperfect ground truth:** the host-profile detector demonstrably flags
  attack behavior (the Infiltration victim's lateral scanning) that the
  corrected labels still call benign — recall/FPR are bounded by label quality.
- **English-only CTI:** the technique mapper is evaluated only on English TRAM
  sentences; non-English reports are out of scope.
- **Sub-technique confusions:** mapper sub-technique hit@1 is 0.277; parent
  granularity (0.690 @5) is the reliable level. TRAM sentences are short and
  often context-free, so single-sentence scores are a floor — campaign-level
  aggregation (`aggregate_matches`) is worth roughly +26% F1 in multi-report
  settings.

## Ethical considerations & data provenance

All data sources are free and public; no proprietary or paid data is used
(zero-cost rule). Outputs are decision-support only and must not be used to
take automated enforcement action against individuals or hosts.

| Source | Used for | Notes |
|---|---|---|
| [NVD CVE API](https://nvd.nist.gov/developers/vulnerabilities) | OSINT ingestion | U.S. government public data |
| [CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) | OSINT ingestion | U.S. government public data |
| [MITRE ATT&CK](https://attack.mitre.org/) | technique catalog + procedures | © The MITRE Corporation, used per ATT&CK terms of use |
| [AlienVault OTX](https://otx.alienvault.com/) | OSINT ingestion | community threat feed |
| [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | IDS train/eval | corrected variant, Engelen et al., WTMC 2021 |
| [TRAM](https://github.com/center-for-threat-informed-defense/tram) | mapper benchmark | analyst-annotated, Center for Threat-Informed Defense |

Models used zero-shot or trained only on the public datasets above; no
personal data is collected. CIC-IDS2017 is a synthetic testbed capture, not
production traffic from real users.

## Caveats for deployment

- **Re-calibrate thresholds on the target network's benign traffic** before
  use. Every operating point here is calibrated on CIC-IDS2017 Mon–Wed benign
  flows and drifts under distribution shift; thresholds will not transfer.
- **Treat ground-truth labels as imperfect.** Recall/FPR figures are bounded by
  CIC-IDS2017 label quality, which provably mislabels at least the Infiltration
  victim's scanning as benign.
- **Run the detectors as an ensemble.** They are complementary by construction
  — supervised (seen families ≈ 1.0), autoencoder (Infiltration/DDoS),
  sequence (web attacks), host-profile (PortScan) — and the replay service runs
  all of them. No single model is sufficient.
- **Keep a human in the loop.** Mapper suggestions and IDS alerts are ranked
  candidates for analyst review, not verdicts.
