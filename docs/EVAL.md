# Model evaluations

## IDS baseline — LightGBM on corrected CIC-IDS2017

Trained on the corrected dataset (Engelen et al., WTMC 2021 — >20% of original
flows relabeled/fixed), "Attempted" flows dropped, identifier/topology columns
(IPs, ports, timestamps) excluded to prevent testbed shortcut learning.
Reproduce: `python -m sentinel.ids.train [--split temporal]`. Tracked in
MLflow (`ids-lightgbm-baseline`).

| Run | ROC-AUC | PR-AUC | F1 | FPR |
|---|---|---|---|---|
| Random 80/20 split (full 2.2M flows) | 0.9998 | 0.9994 | 0.988 @0.5 | 0.4% |
| Temporal split (train Mon–Wed, test Thu–Fri) | 0.9895 | 0.972 | **0.001** @0.5 | 0.00% |
| Temporal split, benign-calibrated threshold (`--calibrate-fpr 0.01`) | 0.9895 | 0.972 | **0.800** | 1.5% |

The calibrated row completes the story: ranking transfers to unseen attack
families far better than the default threshold suggests (DDoS and all three
web attacks 0.94–1.00 recall, PortScan 0.51, Bot 0.00) — the famous temporal
"collapse" is mostly threshold miscalibration, fixable from benign traffic
alone, no attack labels needed.

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

## Anomaly detector — benign-only autoencoder, same temporal split

Torch MLP autoencoder trained on Mon–Wed **benign flows only** (no attack
labels), alerting when reconstruction error exceeds the 99th percentile of
held-out benign error. Reproduce: `python -m sentinel.ids.anomaly`. Tracked
in MLflow (`ids-autoencoder`).

Per-family recall on Thu–Fri attacks, side by side with the supervised
baseline under the identical temporal split:

| Attack family (unseen in training) | LightGBM @0.5 | Autoencoder @p99 |
|---|---|---|
| Infiltration | 0.000 | **0.844** |
| DDoS | 0.001 | **0.705** |
| Web Attack – XSS | 0.000 | **0.667** |
| Web Attack – Brute Force | 0.000 | **0.477** |
| Web Attack – SQL Injection | 0.000 | 0.000 |
| Bot | 0.000 | 0.060 |
| PortScan | 0.001 | 0.007 |
| **Overall recall / FPR** | ~0.000 / 0.00% | **0.268** / 6.3% |

### Backend: MLX vs torch-MPS (10 seeds, full benign Mon–Wed train set)

The autoencoder has two interchangeable backends; the MLX port
(`sentinel/ids/anomaly_mlx.py`, identical architecture/protocol) was adopted
as the auto-selected default on Apple silicon after a multi-seed benchmark
(`python scripts/bench_anomaly.py --seeds 10`):

| backend | train (s) | score (s) | ROC-AUC | recall@p99 | FPR@p99 |
|---|---|---|---|---|---|
| torch-MPS | 3.93 ± 0.20 | 0.155 | 0.915 ± 0.008 | 0.245 ± 0.050 | 0.046 |
| MLX | **1.18 ± 0.03** | **0.102** | 0.913 ± 0.015 | 0.275 ± 0.069 | 0.037 |

Metric parity confirmed at 10 seeds (MLX nominally ahead on recall);
training is 3.3× faster with far lower run-to-run variance. MLX
also links no OpenMP, so it can share a process with LightGBM — the torch
backend deadlocks there (duplicate libomp on macOS, in either import order),
which repeatedly hung the replay service before the switch. torch remains the
fallback (and the only backend on Linux CI/Docker).

## Sequence model — per-host flow streams (MLX gated recurrence)

Hypothesis: attacks whose individual flows look benign (port scans, bot
beacons) are anomalous as a *sequence from one host*. An input-gated
recurrent cell predicts each next flow from the host's preceding window
(16 flows, benign Mon–Wed training only); window score = next-step
prediction error. Reproduce: `python -m sentinel.ids.sequence`.

Three iterations, all recorded (FPR ≈ 2%, window labeled by its last flow):

| Family | error-only | + two-sided | + inter-arrival Δt |
|---|---|---|---|
| Web Attack – XSS | **1.000** | 1.000 | **1.000** |
| Web Attack – Brute Force | **0.941** | 0.941 | 0.824 |
| DDoS | 0.021 | 0.021 | **0.486** |
| Infiltration | 0.333 | 0.333 | 0.333 |
| PortScan / Bot / SQLi | 0.000 | 0.000 | 0.000 |

Findings, kept honest:

- **Genuine unique adds** over the per-flow autoencoder: XSS 1.00 (vs 0.67)
  and Web Brute Force 0.82–0.94 (vs 0.48) — web attacks are sequence-irregular.
- Per-host **inter-arrival time** (leakage-free: relative, not absolute)
  raised DDoS from 0.02 to 0.49 — flood cadence is a timing signature.
- **Seed stability & window size** (5 seeds at w16, 3 at w32): XSS recall is
  1.000 in every run. Window 32 dominates window 16 on every family — DDoS
  0.60 ± 0.06 (vs 0.50 ± 0.04), Brute Force 0.96 ± 0.06 (vs unstable at w16,
  one seed collapsing to 0.0) — and is now the default (`--window 32
  --stride 16`).
- **Negative result**: scans/beacons stayed at 0.000 across all three
  variants. Their windows are *more predictable than benign traffic*
  (ROC-AUC ≈ 0.42, i.e. inverted), and not even extreme on the low side —
  prediction-error magnitude cannot represent "suspiciously machine-like".
  Detecting them likely needs explicit fan-out/cardinality features
  (distinct destination ports/hosts per window), noted as future work.

The three detectors cover different families, so the replay service runs all
of them; per-family best: supervised (seen families ≈ 1.0), autoencoder
(Infiltration 0.84, DDoS 0.71), sequence model (XSS 1.00, Brute Force 0.94).

## Host-profile detector — fan-out statistics (no neural net)

Follow-up to the sequence model's scan/beacon negative. Four explainable
per-window statistics over each host's stream (unique dst ports, unique dst
IPs, log flow rate, log mean packet count), robust-scaled on benign Mon–Wed
windows, alerting on excess max robust-z. Destinations feed *counts* only —
never raw feature values. Reproduce: `python -m sentinel.ids.profile`.

| Metric | Value |
|---|---|
| PortScan recall | **0.998** (was 0.000 for every other detector) |
| Nominal FPR | 12.1% |
| FPR excluding host 192.168.10.8 | **1.15%** |
| Bot recall | 0.000 (open — needs a periodicity signature) |

The nominal FPR is not noise: **90% of the "false positives" trace to one
host, 192.168.10.8, on Thursday** — the documented Infiltration victim, which
port-scans the internal subnet after the meterpreter compromise. Those flows
are labeled BENIGN in the corrected ground truth; the detector flags the
victim's lateral scanning that the labels miss. With that host excluded the
detector operates at its calibrated ~1% FPR. Diagnostic trail: threshold
tightening (p99.9) and a tiny-flow side-condition both failed to move FPR
(saturated windows are also small-flow), host attribution found the single
responsible source.

### Beacon (Bot) follow-up: periodicity statistics

Two beacon signatures were added (inter-arrival coefficient-of-variation →
periodicity score; repeated-destination ratio) and tested in two stream
groupings:

| Grouping | Bot recall | FPR | window ROC-AUC |
|---|---|---|---|
| per-host (deployed) | 0.000 | — | 0.79 |
| per-(src→dst) pair (`--group-by pair`) | **0.056** | 0.9% | 0.83 |

Per-host windows interleave the victim's benign traffic with its beacons,
destroying the timer pattern; isolating each (src→dst) channel finally puts
Bot above zero — the first detector to do so — but only the extreme tail
clears the benign-calibrated threshold (the ARES beacon's cadence overlaps
benign periodic services like NTP). Recorded as a foothold, not a solution:
ranking signal exists (AUC 0.83), the operating-point recall does not yet.
Host grouping stays the deployed default (PortScan 0.998).

### Spectral beacon detector — Schuster periodogram (third beacon attempt)

A proper spectral test replaces the variance heuristic: per (src→dst) channel,
the windowed Schuster statistic max_P mean|Σ exp(2πi t/P)|²/n scores how
tightly events lock to a period. Building it surfaced three real pitfalls
(all now in the code): second-quantized timestamps are perfectly periodic at
P=1s (grid floored at 4s); max-over-periods inflates the noise floor (averaged
coherence at a fixed period instead); and **two ground-truth gaps** — 67% of
Bot flows carry the "- Attempted" label (payload-less C2 polling, which *is*
the beacon), and the ARES victims keep beaconing past the labeled window so
channels are labeled by dominant attack, not last flow.

Result on the real C2 channels (5 hosts → 205.174.165.73): ROC-AUC **0.73**,
recall@p99 **0.000**. The C2 beacons score 12.8–15.6 of a possible 16 — but
benign p99 already saturates at 16.0, because **benign machine-periodic
services (NTP, keepalives) are more spectrally coherent than a jittered
attacker beacon**. The ranking signal is real; the operating point is not,
and tightening the threshold cannot separate sub-maximal beacons from maximal
benign timers.

Conclusion across three attempts (variance pairs 0.056, spectral AUC 0.73,
both recall ~0): **beacon detection by periodicity alone is the wrong frame in
this dataset** — perfect periodicity is benign infrastructure. A real beacon
hunter needs destination reputation / JA3 / payload-size entropy, none of
which CIC-IDS2017 flow features expose. Kept as a characterized negative; the
detector ships behind `make ids-spectral` for the documented C2-channel
ranking, not as an ensemble member.

The two models are complementary by construction: the supervised model is
near-perfect on attack families it has seen (random-split table above), the
autoencoder catches a meaningful share of families it has never seen — which
is the scenario that matters for a fusion platform. Honest caveats, kept on
purpose: observed FPR (6.3%) drifts above the calibrated 1% because Thu–Fri
benign traffic differs from Mon–Wed benign traffic (distribution shift), and
low-rate scans/botnet beacons reconstruct too well to alert (they look like
small normal flows). Ensemble + campaign-context fusion is the next layer.

# Technique mapper evaluation

Zero-shot mapping of CTI sentences to ATT&CK techniques, evaluated against
[TRAM](https://github.com/center-for-threat-informed-defense/tram) bootstrap
data (11,130 sentences hand-labeled by analysts, 50 techniques). The mapper
retrieves against the **full enterprise catalog (697 active techniques)** with
no task-specific training — unlike classifier approaches (TRAM itself,
TTPxHunter) that are trained on and restricted to the ~50 most common
techniques.

Reproduce: `python scripts/eval_mapper.py --sample 2000` (seed 13).

## Full corpus (10,411 labeled sentences after the ≥4-word filter)

hit@k = a gold technique appears in the top-k predictions; parent = credit at
parent-technique level (T1059.001 → T1059). Bi-encoder retrieval
(SecureBERT2.0-biencoder) vs top-20 candidates reranked pairwise with the
cross-encoder (`--sample 12000 [--rerank]`).

| k | retrieval hit@k / parent | + rerank hit@k / parent |
|---|---|---|
| 1 | 0.216 / 0.324 | **0.277 / 0.401** |
| 3 | 0.366 / 0.498 | **0.465 / 0.589** |
| 5 | 0.446 / 0.583 | **0.543 / 0.663** |
| 10 | 0.549 / 0.683 | **0.620 / 0.739** |

Reranking is worth +8–10 pp on every metric at full-corpus scale (earlier
2,000/300-sentence samples were consistent with these numbers). Cost:
retrieval ~12 ms/sentence, reranking ~1.6 s/sentence on Apple silicon —
reranking suits report-level ingestion, retrieval-only suits interactive use.

## Mapper v2: hybrid lexical+dense retrieval and procedure enrichment

Two retrieval upgrades A/B-tested on the full corpus (`--procedures`,
`--hybrid`): technique docs enriched with real ATT&CK *procedure examples*
("APT29 used ..."), and BM25 fused with the dense ranks via reciprocal-rank
fusion (~60 lines, no new dependencies).

| Config | hit@5 / parent@5 | hit@10 / parent@10 |
|---|---|---|
| dense only (baseline) | 0.446 / 0.583 | 0.549 / 0.683 |
| + procedures only | 0.392 / 0.542 | 0.508 / 0.648 |
| + hybrid only | 0.520 / 0.665 | 0.638 / 0.766 |
| **+ procedures + hybrid (adopted)** | **0.553 / 0.690** | **0.682 / 0.792** |
| (reference: dense + cross-encoder rerank) | 0.543 / 0.663 | 0.620 / 0.739 |

Findings: procedures *hurt* the dense embedding alone (long appended text
dilutes it) but are the best config combined with BM25 — the lexical side is
what exploits the procedure vocabulary (tool names, commands, registry
paths). The adopted config **beats the 130× more expensive cross-encoder
rerank** at bi-encoder cost. Design note: the mapper ranks by fusion but
reports the dense cosine, because RRF scores are rank-based and carry no
absolute confidence — downstream tagging thresholds stay on the cosine scale.

Notes:

- Single-sentence scores are the floor: SENTINEL corroborates evidence across
  the sentences/reports of a campaign (`aggregate_matches`), which is worth
  roughly +26% F1 in multi-report settings (arXiv:2604.07470).
- TRAM sentences are short and often context-free ("network traffic
  communicates over a raw socket"), so exact sub-technique hit@1 understates
  document-level usefulness; parent-level hit@5 (0.58) is the better proxy for
  the triage use case (suggest candidate techniques to an analyst).

## Conformal thresholds and label-free alert-budget control

Adapted from the author's time-series conformal toolkit with two NIDS-specific
changes: one-sided split-conformal p-values against benign calibration scores
(finite-sample FPR guarantee under exchangeability), and a label-free online
controller that regulates the **alert rate** — ACI's update rule with the
interval-miss signal replaced by the alert indicator, since live NIDS has no
ground truth. Reproduce: `python scripts/eval_conformal.py` (autoencoder
scores, Thu–Fri processed chronologically).

| Policy | alert rate | FPR | recall DDoS / Infiltration / XSS |
|---|---|---|---|
| static p99 | 10.7% | 5.96% | 0.60 / 0.84 / 0.67 |
| conformal p ≤ 0.01 | 10.7% | 5.96% | 0.60 / 0.84 / 0.67 |
| **budget controller (α=1%)** | **0.99%** | **1.10%** | 0.008 / **0.84** / **0.70** |

Findings:

- Conformal p-values match the static threshold numerically — the value is
  the *guarantee*, and the 6% realized FPR is the proof that exchangeability
  breaks across the Mon–Wed → Thu–Fri shift.
- The budget controller holds the alert rate at its target through the same
  drift — bounded analyst load with no labels — and rare attacks keep
  alerting (Infiltration unchanged, XSS up).
- The honest trade-off: a 1% budget cannot represent a volumetric attack
  (DDoS is ~10% of the stream), so flood recall collapses by construction.
  Budget control is a rare-event/analyst-load instrument; volumetric floods
  are covered by the supervised and profile detectors in the ensemble.
- Caveat recorded: folding non-alerting scores into the controller's memory
  tracks benign drift but can be self-poisoned by high-volume sub-threshold
  attacks; `adapt_memory=False` trades adaptivity for immunity.

## Temporal intelligence analytics (CTI side)

Three graph-derived analytics close the loop from raw ingestion to an analyst
handoff, all computed from report timestamps with no extra storage:

- **Trending techniques** (`/trending`): mention-rate lift, recent window vs
  prior, +1-smoothed. On the live feed the Nx supply-chain story surfaces as
  T1195.001 (Compromise Software Dependencies) at lift x8.
- **Feed drift** (`/feed-drift`): Population Stability Index over the report
  *source* mix — the CTI analogue of the IDS benign-drift the conformal
  controller handles. The standard PSI bands (<0.1 stable, <0.25 moderate,
  else significant) flag when feed composition shifts.
- **Daily briefing** (`/briefing`): plain-text SOC handoff fusing trending
  techniques, KEV-weighted campaign counts, and drift verdict.

## Cross-dataset generalization: 2017 → 2018 (the headline honesty test)

The question within-dataset numbers cannot answer: does a model trained on
CIC-IDS2017 detect the *same attack* on a *different network*? Brute-force is
the probe — 2017's FTP/SSH-Patator and 2018's FTP/SSH-BruteForce are the same
attack under different label strings. A canonical-name normalizer
(`sentinel/ids/cross_dataset.py`) aligns the two datasets' divergent column
conventions ("Total Fwd Packet" vs "Tot Fwd Pkts") to **65 shared flow
features**; the model trains on 2017 and tests on a real CSE-CIC-IDS2018 day
(1.05M flows from the AWS open-data bucket). Reproduce:
`python scripts/eval_cross_dataset.py`.

| Metric | Value |
|---|---|
| within-2017 ROC-AUC | **1.0000** |
| cross-2018 ROC-AUC | 0.940 |
| cross-2018 **recall @ 1% FPR** | **0.000** |

The single most important number in the project. Within-dataset brute-force is
*perfectly* separable (AUC 1.0); the model still **ranks** 2018 attacks above
2018 benign (AUC 0.94), but at the deployed operating point it detects **none
of them** — the absolute threshold learned on 2017 lands in the wrong place
for 2018's score distribution. This is the published within-dataset-inflation
failure (arXiv:2402.10974) reproduced first-hand, and it is the empirical
justification for the conformal **budget controller** above: the fix for a
threshold that doesn't transfer is to re-derive it from the target network's
own benign traffic, online and label-free — exactly what the controller does.
The cross-dataset experiment states the problem; the conformal controller is
the answer *within a network* — but not across one, and we measured the
limit rather than assuming it (`scripts/eval_conformal_cross.py`):

| Policy on the 2018 stream | recall | FPR |
|---|---|---|
| static threshold (from 2017 benign) | 1.000 | 23.5% |
| recalibrated on 2018 benign (label-free) | 0.000 | 0.2% |

Re-deriving the threshold from the target network's own benign traffic
controls the false-alarm rate but **cannot recover detection** on 2018: the
transferred model's benign and attack scores overlap so heavily that no
label-free threshold separates them (catch everything at 23% FPR, or nothing
at 0.2%). The honest conclusion across both experiments: label-free
recalibration recovers rare-attack recall under *within-network* drift
(Infiltration 0.84, XSS 0.70 on the temporal split) but is not a substitute
for target-domain adaptation across *different* networks. Cross-network
transfer needs labels or feature adaptation — recalibration alone is not
enough, and the project says so because it ran the test. Together they are the project's thesis: *report the number that
survives a network change, and build the mechanism that makes it survivable.*