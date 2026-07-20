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

Conclusion across three periodicity attempts (variance pairs 0.056, spectral
AUC 0.73, both recall ~0): **beacon detection by periodicity alone is the wrong
frame in this dataset** — perfect periodicity is benign infrastructure. The
spectral detector ships behind `make ids-spectral` for the documented C2-channel
ranking, kept as a characterized negative.

### Beacon by data-size dispersion — closes the CIC Bot gap, but ARES-specific

Re-framing closed the *CIC-IDS2017* gap (it does not generalize — see below). An
ARES C2 channel interleaves payload-less poll
flows ("Bot - Attempted" — 67% of 2017 Bot flows) with data-carrying tasking
flows, so a single (src→dst) channel's forward-payload **sizes are wildly
dispersed**; a benign periodic service (NTP) sends a uniform packet every time,
so its sizes barely move. Scoring each channel by the coefficient of variation
of its forward bytes and mean packet length — benign-calibrated, max robust-z,
thresholded at the benign-channel p99 — is the first detector to convert the
beacon ranking signal into operating-point recall. Reproduce: `make ids-beacon`.

| Frame | Bot channel recall @~1% FPR | channel ROC-AUC |
|---|---|---|
| periodicity (variance pairs / spectral) | 0.000–0.056 | 0.73–0.83 |
| **data-size dispersion (`ids/beacon.py`)** | **1.000 (5/5)** | **0.995** |

The dispersion statistics are behavioral (size CV), never an IP/port/timestamp,
so the no-topology-leak rule holds. The 2018 Bot capture (286k flows) confirms the
*mechanism* is not a 2017 fluke: **50.3% empty polls ≤1 byte + 49.7% data flows
>100 bytes** — exactly the bimodality that produces high per-channel variance.

#### …but it does not generalize beyond ARES (CTU-13, the honest negative)

The caveat at the time was n=5 (all five 2017 C2 channels go to one ARES server),
so the detector was cross-validated on **CTU-13** (Stratosphere): 13 captures,
**seven botnet families, 1,470 botnet channels with real IPs** — the dataset the
foothold needed. Same detector, per-scenario benign calibration (`make
eval-beacon-ctu13`). It **fails to generalize**:

| | recall @~1% FPR | note |
|---|---|---|
| CIC-IDS2017 (ARES) | 1.000 (5/5) | the bimodal poll+tasking signature |
| **CTU-13 (7 families, 1,470 ch.)** | **0.010 (15/1,470)** | chance; most family AUCs **< 0.5** |

Only Rbot scenarios 10–11 show signal (0.71/1.00); every other family is ~0, and
robust to using `SrcBytes` (forward analog) instead of `TotBytes`. The AUCs are
*inverted* (botnet channels are **more uniform** than diverse benign traffic) —
which explains it: ARES is bimodal (high dispersion), but Rbot/Neris/Virut beacon
with **uniform** payload sizes (low dispersion). Data-size dispersion is one
specific C2 signature, **not a universal beacon detector**. (The irony: the
regular-cadence botnets dispersion misses are the ones the abandoned *periodicity*
detector might catch — no single statistic covers all botnets.)

**Honest status:** the beacon detector genuinely closed the CIC-IDS2017 Bot gap
and stays a CIC ensemble member, but it is **ARES-specific** — a characterized
result on one botnet, not a general capability. Validation turned a "foothold"
into a documented limitation, which is the point of running it.

The two models are complementary by construction: the supervised model is
near-perfect on attack families it has seen (random-split table above), the
autoencoder catches a meaningful share of families it has never seen — which
is the scenario that matters for a fusion platform. Honest caveats, kept on
purpose: observed FPR (6.3%) drifts above the calibrated 1% because Thu–Fri
benign traffic differs from Mon–Wed benign traffic (distribution shift), and
low-rate scans/botnet beacons reconstruct too well to alert (they look like
small normal flows). Ensemble + campaign-context fusion is the next layer.

## Application-layer SQLi detection — a different modality (`make sqli`)

CIC-IDS2017 has only **12 SQLi flows, none in training**. On the basic
volume/timing features they're indistinguishable from benign HTTP (max robust-z
over duration/bytes/packets ≈ **1.0**), so the **unsupervised** flow detectors —
autoencoder, sequence, profile — score SQLi **0.0**: reconstruction- and
prediction-error simply can't represent it. (A *calibrated supervised* model is
the exception: using the full 70-feature set it flags all 12 SQLi flows at recall
1.0 / 1.5% FPR — but that rests on 12 within-dataset flows and only says
"attack-ish," not "SQLi.") What you actually want for SQLi is a detector that
recognizes the attack **by its signature** and works on real requests — the SQL
string lives in the request *payload*, which CICFlowMeter never captures. So SQLi
gets a different modality: a character n-gram (TF-IDF `char_wb` 1–3) +
logistic-regression classifier over request payloads, the application-layer / WAF
analogue of the flow IDS, mapped to T1190. Code: `src/sentinel/ids/sqli.py`.

Validated the SENTINEL way — **cross-corpus** (train one public payload source,
test a different one), so the number reflects generalization, not memorization:

| Eval | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| within-corpus (avg of 2 sources, 3 seeds) | 1.000 | 1.000 | 0.995 | 0.997 |
| cross: HttpParams → Kaggle SQLiV2 | 0.997 | 1.000 | 0.969 | 0.984 |
| cross: Kaggle SQLiV2 → HttpParams | 1.000 | 1.000 | 0.997 | 0.998 |

Char n-grams carry the generalization: they key on SQL syntax (quotes, comment
markers, `union select`, `or 1=1`) that survives across payload styles, where
word tokens would overfit one corpus's vocabulary. Corpora are free and public
(Morzeux HttpParamsDataset; Kaggle SQLiV2), cached under `data/sqli/`. Why this
beats the calibrated-supervised flow result despite the same headline recall: it
recognizes SQLi *specifically* (not "anomalous flow"), generalizes across
independent corpora (so it isn't memorizing 12 testbed flows or relying on a
within-dataset threshold), and runs on the actual HTTP request — a deployable WAF
signal. Honest scope: it inspects payloads, not netflow — it complements the flow
ensemble rather than fixing it. It is wired into the platform via a **WAF replay**
(`make waf-replay`, `ids/waf_replay.py`): the detector scores an HTTP-request
stream and persists flagged requests as `model="sqli"` Alerts tagged T1190, which
fuse with T1190 campaigns and surface in the host rollup like any flow detection.
Like the flow replay over the CIC *testbed*, this is a replay over a labelled
corpus (the public payload sets carry no client IPs, so requests are attributed
to synthetic RFC 5737 documentation IPs — clearly not real attribution).

## Ensemble coverage — the unit you actually deploy (`make eval-ensemble`)

Judging the platform on any single model's overall recall is the wrong unit — the
autoencoder's **0.268 overall** is modest because it is unsupervised on *unseen*
families, but you deploy the **five-detector ensemble**, not the autoencoder
alone. Run on the same temporal split (train Mon–Wed, test the unseen Thu–Fri
families) at ~1% benign FPR, each family is carried by its specialist:

| Unseen family | Best detector | Recall | (no single model covers all) |
|---|---|---|---|
| DDoS | supervised (calibrated) | 1.000 | autoencoder 0.70, sequence 0.64 |
| Infiltration | supervised | 0.938 | autoencoder 0.84 |
| PortScan | profile / beacon | 1.000 | supervised 0.51, autoencoder 0.01 |
| Bot | beacon | 1.000 | supervised 0.00, autoencoder 0.06 |
| Web Brute Force | sequence / supervised | 1.000 | autoencoder 0.48 |
| Web XSS | sequence / supervised | 1.000 | autoencoder 0.67 |
| Web SQL Injection | supervised¹ | 1.000 | unsupervised 0.00; payload detector 0.98 cross-corpus |

**7/7 unseen families covered at recall ≥ 0.93 by their specialist**, where the
strongest single unsupervised model (the autoencoder) averages 0.268. That is the
honest case for the ensemble: the models are complementary by construction, and
the system catches what no one detector can. The combined alert rate is higher
than any single detector's (union of five ~1% operating points) — the standard
ensemble trade. ¹SQLi-via-supervised rests on 12 within-dataset flows; the payload
detector is the robust, SQLi-specific answer (above).

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
ground truth. Reproduce: `make eval-conformal` (autoencoder scores, Thu–Fri
processed chronologically).

| Policy | alert rate | FPR | recall DDoS / Infiltration / XSS |
|---|---|---|---|
| static p99 | 10.7% | 5.96% | 0.60 / 0.84 / 0.67 |
| conformal p ≤ 0.01 | 10.7% | 5.96% | 0.60 / 0.84 / 0.67 |
| **budget controller (α=1%)** | **0.99%** | **1.10%** | 0.008 / **0.84** / **0.70** |

The controller is not just an offline study: the flow-replay service can gate its
one-sided anomaly detectors (autoencoder, profile) through it with `python -m
sentinel.ids.replay --conformal` (sequence is two-sided and beacon is a static
per-channel set, so both keep the percentile). The default stays the fixed benign
percentile so the recorded ensemble numbers reproduce; `--conformal` is the
drift-robust operating mode.

### Does enabling `--conformal` cost ensemble coverage? (`make eval-ensemble --conformal`)

The budget controller is opt-in because it trades a detector's *own* recall for a
bounded alert rate. The question for adoption is whether that costs the
**ensemble** anything — and it doesn't, because the budget-capped detectors
(autoencoder, profile) are not the specialists for the families a 1% rate cap
suppresses. Same temporal split, best detector per unseen family, percentile vs
conformal:

| Unseen family | best specialist (recall, unchanged) | autoencoder p99 → conformal | profile p99 → conformal |
|---|---|---|---|
| DDoS | supervised **1.000** | 0.70 → 0.04 | 0.00 → 0.00 |
| Infiltration | supervised **0.938** | 0.84 → 0.62 | 0.33 → 0.00 |
| PortScan | beacon **1.000** | 0.01 → 0.00 | 1.00 → 0.01 |
| Bot | beacon **1.000** | 0.06 → 0.03 | 0.00 → 0.04 |
| Web Brute Force | sequence/supervised **1.000** | 0.48 → 0.48 | 0.00 |
| Web XSS | sequence/supervised **1.000** | 0.67 → 0.67 | 0.00 |
| Web SQL Injection | supervised **1.000** | 0.00 | 0.00 |

**7/7 unseen families stay covered at recall ≥ 0.93 under `--conformal`.** The cap
visibly lowers the gated detectors' own recall — most where a 1% budget cannot
represent a high-prevalence attack (autoencoder DDoS 0.70→0.04, profile PortScan
1.00→0.01); sparse families it barely touches (XSS 0.67 unchanged). But every
family's specialist is a *percentile / supervised* detector (supervised DDoS 1.0,
beacon PortScan/Bot 1.0), so the ensemble loses nothing. The empirical case for
`--conformal` as a safe operating mode: it buys the drift-robust **1.10% FPR (vs
the static p99's 5.96%)** at **no cost to ensemble coverage**.

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
handoff, all computed from report timestamps with no extra storage. Windows key
off each report's **publish date** (falling back to ingest time), so a one-shot
bulk ingest of feeds whose items span weeks still yields a real timeline rather
than collapsing everything into "now".

- **Trending techniques** (`/trending`): mention-rate lift, recent window vs
  prior, +1-smoothed. On the live multi-source feed (17 keyless CTI publishers)
  the surging stories surface honestly — e.g. T1588.007 (AI) at lift x7 and
  T1584.005 (Botnet) at x4 in a recent run.
- **Feed drift** (`/feed-drift`): Population Stability Index over the report
  *source* mix — the CTI analogue of the IDS benign-drift the conformal
  controller handles. The standard PSI bands (<0.1 stable, <0.25 moderate,
  else significant) flag when feed composition shifts. Counts are additively
  (Jeffreys, α=0.5) smoothed over the union of sources before the index is
  computed, so a source that appears or vanishes between windows yields a
  finite, comparable PSI instead of the unbounded blow-up a zero-bin log gives.
- **Daily briefing** (`/briefing`): plain-text SOC handoff fusing trending
  techniques, KEV-weighted campaign counts, and drift verdict.

## Fusion scoring: from shared tag to calibrated correlation

The platform is named for fusion, so the join between an IDS alert and a CTI
campaign has to be more than set overlap. Naive overlap ("the alert and the
campaign both mention T1499") treats a ubiquitous technique exactly like a rare,
specific one — the precise objection a reviewer raises: *lots of campaigns
involve DoS, so why is this match meaningful?* `correlate/fusion.py` answers it
by scoring every match on three independent, interpretable signals and combining
them as a geometric mean (conjunctive — a weak factor drags the whole down):

- **specificity** — min-max-normalized IDF of the shared technique over the
  report corpus. A technique in 1-of-N reports is surprising (→1.0); one in most
  reports carries no discriminating signal (→0.0).
- **recency** — exponential decay on the matched campaign's age (latest member
  report), half-life 30 days (`SENTINEL_FUSION_RECENCY_HALF_LIFE_DAYS`). A live
  correlation outranks a months-old one.
- **corroboration** — the campaign's own mean technique score discounted by a
  saturating function of its member-report count.

**Worked example** (`tests/test_fusion.py`). Two campaigns each share one
technique with the same alert. `camp:rare` shares T1195.001 (supply-chain,
1-of-5 corpus reports, reported today, 3 corroborating reports at score 0.8);
`camp:common` shares T1110 (brute force, 5-of-5 reports, last seen 120 days ago,
1 report at score 0.4). Under naive overlap both are equal "matches". Scored:

| campaign | specificity | recency | corroboration | **strength** |
|----------|-------------|---------|---------------|--------------|
| camp:rare   | 1.000 | 1.000 | 0.700 | **0.888** |
| camp:common | 0.000 | 0.062 | 0.200 | **0.000** |

The generic match collapses to zero here because T1110 saturates the corpus
(every report carries it → IDF rarity 0). In a real feed a common tag is
small-but-positive rather than exactly zero, but the ordering is the point: the
specific, recent, corroborated correlation is surfaced and the coincidental one
is suppressed. The same strength scales the per-host risk bonus
(`correlate/hosts.py`) and is returned component-by-component on
`/alerts/{id}/context` and `/hosts`, so the dashboard shows *why* a correlation
ranks where it does rather than an opaque confidence.

Matching is at the **ATT&CK family level**: the IDS attack map emits parent
techniques (DoS → T1499) while the NLP tagger tags sub-techniques (T1499.004),
so a sub-technique fuses with its parent (standard ATT&CK roll-up). On the live
graph this is the difference between a DoS alert silently failing to correlate
and surfacing its DoS campaign — fused hosts rose 4 → 12 when family matching
replaced exact-string overlap, the specificity still computed on the campaign's
actual (sub-)technique IDF.

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
enough, and the project says so because it ran the test.

### Can we beat it? Five fixes, measured (`scripts/eval_domain_adapt.py`)

Recall at a target-benign-calibrated 1% FPR, 2017 → 2018. The quantile rows
are mean ± standard deviation over three complete split/refit runs (seeds
13–15); the other rows are the seed-13 controls. This is the full-data
domain-adaptation harness, so its single-seed controls need not exactly match
the separately sampled cross-family study below. For the quantile rows, the
target transform and raw-imputation medians fit on adaptation-pool benign flows
only. The legacy controls' shared imputation median fits on the whole disjoint
pool (including attacks), while thresholds fit on calibration-pool benign
flows. Held-out test flows are transformed and scored, never used to fit any
of these statistics.

| Family | Fix | recall | FPR | AUC |
|---|---|---:|---:|---:|
| brute-force | baseline (train 2017) | 0.000 | 0.002 | 0.927 |
| brute-force | CORAL covariance alignment | 0.000 | 0.000 | 0.555 |
| brute-force | transfer-stable features | 0.000 | 0.007 | 0.009 |
| brute-force | target-trained autoencoder | 0.000 | 0.010 | 0.718 |
| brute-force | quantile map (source → target transport) | 0.169 ± 0.239 | 0.003 ± 0.004 | 0.525 ± 0.171 |
| brute-force | **benign quantile space** | **0.502 ± 0.355** | **0.005 ± 0.001** | **0.989 ± 0.006** |
| brute-force | **few-shot: +50 labelled 2018 flows** | **1.000** | **0.005** | **1.000** |
| DoS | baseline (train 2017) | 0.016 | 0.005 | 0.597 |
| DoS | CORAL covariance alignment | 0.087 | 0.002 | 0.543 |
| DoS | transfer-stable features | 0.000 | 0.000 | 0.541 |
| DoS | target-trained autoencoder | 0.061 | 0.010 | 0.910 |
| DoS | quantile map (source → target transport) | 0.000 ± 0.000 | 0.005 ± 0.004 | 0.454 ± 0.050 |
| DoS | benign quantile space | 0.033 ± 0.022 | 0.005 ± 0.001 | 0.523 ± 0.055 |
| DoS | **few-shot: +50 labelled 2018 flows** | **0.839** | **0.001** | **0.995** |
| Bot | baseline (train 2017) | 0.000 | 0.004 | 0.684 |
| Bot | CORAL covariance alignment | 0.000 | 0.000 | 0.712 |
| Bot | transfer-stable features | 0.000 | 0.009 | 0.284 |
| Bot | target-trained autoencoder | 0.001 | 0.010 | 0.244 |
| Bot | quantile map (source → target transport) | 0.000 ± 0.000 | 0.008 ± 0.002 | 0.242 ± 0.050 |
| Bot | benign quantile space | 0.001 ± 0.001 | 0.007 ± 0.001 | 0.635 ± 0.051 |
| Bot | **few-shot: +50 labelled 2018 flows** | **0.993** | **0.010** | **0.998** |

The fifth approach is **benign quantile alignment**: represent each feature by
its percentile against that network's own benign traffic. Its symmetric
quantile-space variant barely clears the pre-registered 0.5 brute-force gate,
but the 0.355 standard deviation makes that result seed-sensitive. More
importantly, the promotion check fails on DoS (0.033) and Bot (0.001).
Source-to-target quantile transport is weaker still. This is a useful partial —
per-feature rank normalization can narrow one shifted boundary — not the first
general working label-free transfer fix.

Methodology note (this matters): the few-shot labels and the test set are a
**disjoint split of the 2018 day** — the model is graded only on flows it
never saw, and the few-shot labels are drawn from a separate pool. An earlier
pass tested on flows that overlapped the few-shot set; the 1.000 survived
removing that contamination, so it is real, not leakage.

**Post-hoc leakage audit (2026-07)** — a "perfect score" earns hostility, so
the 1.000 was re-audited beyond the index-disjoint split:

- **Content-level dedup:** CIC-IDS2018's brute-force day is heavily
  self-similar (only 57% of rows are unique feature vectors; **70% of attack
  test rows are byte-identical duplicates of one of the 25 few-shot attack
  examples** — scripted FTP/SSH logins repeat). An index-disjoint split does
  not neutralize that on its own. Re-scoring after removing every test row
  that exact-duplicates a few-shot row — and, stricter, after dropping all
  duplicates dataset-wide — leaves recall/AUC **unchanged to three decimals**
  (headline exact values: recall 0.99997, AUC 0.99994). The score is not
  duplication leakage.
- **Triviality probe (the honest explanation):** a **depth-1 decision stump**
  trained on only the 50 few-shot rows reaches **AUC 0.997 / recall 1.000**
  on the deduplicated test set, splitting on a single feature
  (`Fwd Seg Size Min > 26`). Brute-force is intrinsically ~one-feature
  separable in-domain; the 2017 model fails only because that feature's scale
  shifts across networks. Read the 1.000 as "a few in-domain labels re-anchor
  a trivially separable boundary", **not** as a general few-shot capability.
  The representative few-shot numbers are DoS 0.955 and Bot ~0.99, whose test
  sets share almost none of brute-force's duplication artifact (0% / 4.7%
  attack-row overlap with the few-shot pool).
- **Minor pre-split leak, fixed in this harness:** NaN-fill medians were computed over the
  whole 2018 day (pool + test) before splitting. One scalar per feature over
  ~1M rows — no material effect (numbers reproduce identically) — but the
  domain-adaptation eval now computes fill medians from the pool only.

None of the label-free methods is a general fix. CORAL and transfer-stable
feature selection remain failures; the target-trained autoencoder can rank
some attacks but cannot reliably clear a usable operating point. Quantile
space is the one narrow exception: it recovers a seed-sensitive mean recall of
0.502 on brute-force, then falls back to 0.033 on DoS and 0.001 on Bot. Only
**few-shot** works consistently across all three families. In the previously
audited brute-force run, 50 labelled target flows recovered 0.99997 recall on
held-out traffic (exact, unrounded: 228,562 of 228,569 attacks alerted, 7
missed; AUC 0.9999436 — earlier drafts printed these at 3 dp as "1.000",
which oversold a near-perfect score as a perfect one).

Why so clean? Not overfitting: FTP/SSH brute-force is intrinsically separable
*once the model has in-domain labels*. The 2017->2018 failure is a
boundary-placement problem (the 2017 boundary lands wrong on 2018's feature
scale); a few target labels re-anchor it. The honest, useful conclusion:
cross-network IDS transfer is a few-shot labelling problem, not a
generally solved representation-alignment problem. The next section tests the
same few-shot recipe across three target attack families rather than trusting
the especially easy brute-force case.

### Cross-family stress test: few-shot is the fix (`scripts/eval_cross_family.py`)

Run across **three different 2018 attack families** on a different network
(brute-force, DoS, Bot), each on a contamination-free held-out split, at a
target-calibrated 1% FPR:

| Family | baseline (2017) | target autoencoder | few-shot 50 | few-shot 500 |
|---|---|---|---|---|
| brute-force | 0.000 (AUC .98) | 0.000 (AUC .67) | **0.99996** | 1.000 |
| DoS | 0.047 (AUC .85) | 0.001 (AUC .84) | **0.955** | 0.995 |
| Bot | 0.000 (AUC **.40**) | 0.000 (AUC .03) | **0.986** | 0.990 |

(Brute-force few-shot-50 exact, unrounded: recall 0.9999556 — 44,998 of 45,000
sampled attacks alerted, 2 missed; AUC 0.9999125. Cells are 3-dp rounded;
scores that round to 1.000 are near-perfect, not literally perfect.)

Two results, both kept honest:

- **The label-free autoencoder does not work.** The hypothesis that a benign-
  only AE trained on target traffic would catch volumetric DoS was *wrong* — it
  ranks DoS reasonably (AUC 0.84) but cannot clear a usable threshold (recall
  0.001). On Bot it is worse than chance; the quantile study above likewise
  produces only a narrow brute-force partial, not a cross-family fix.
- **Few-shot is the fix, and it is robust.** 50 labelled flows of a family
  recover 0.95-0.99 recall on the new network across all three families —
  including Bot, whose 2017 baseline ranks *worse than a coin flip* (AUC 0.40)
  and which 50 labels lift to AUC 0.997. This is not the easy-task artifact of
  a single family; it holds for three distinct attack types on held-out data.

**Conclusion.** Cross-network IDS transfer is a few-shot labelling problem. No
tested label-free transform generalizes: quantile space narrows the gap only
for brute-force and is highly seed-sensitive, while CORAL, feature selection,
quantile transport, and the target-trained autoencoder do not recover a usable
operating point across families. About 50 labelled target flows per family do.
The practical recipe falls out of the ensemble: the unsupervised detectors
(host-profile fan-out, sequence model) surface candidate attacks on the new
network, an analyst confirms ~50, and the supervised model adapts to near-
perfect recall. A handful of labels goes remarkably far. Together they are the project's thesis: *report the number that
survives a network change, and build the mechanism that makes it survivable.*

### Label efficiency and selection — what is deployable? (`scripts/eval_label_efficiency.py`)

The full study compares six selectors across three families, five budgets, and
five seeds. Values are mean ± standard deviation recall at a target-benign-
calibrated 1% FPR; the label pool and test split are disjoint.

- **Balanced random (oracle)** deliberately draws half attack and half benign
  using hidden ground-truth labels. It preserves the earlier controlled budget
  curve, but an operator cannot deploy it before asking for the labels.
- **Random-blind** samples uniformly from the unlabelled pool and is the honest
  deployable random control. Active uses the collapsed source model's
  uncertainty; coreset and cluster use feature geometry; stratified covers ten
  deciles of the blind model's score range.

#### Brute-force

| N | balanced random (oracle) | random-blind | active | coreset | cluster | stratified |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.788 ± 0.076 | 0.798 ± 0.186 | 0.603 ± 0.301 | 0.600 ± 0.300 | 0.801 ± 0.099 | 0.997 ± 0.004 |
| 25 | 0.619 ± 0.460 | 0.949 ± 0.099 | 0.650 ± 0.339 | 0.300 ± 0.368 | 0.957 ± 0.076 | 0.931 ± 0.101 |
| 50 | 0.938 ± 0.123 | 0.992 ± 0.015 | 0.604 ± 0.302 | 0.463 ± 0.352 | 0.748 ± 0.387 | 0.900 ± 0.122 |
| 100 | 1.000 ± 0.000 | 0.898 ± 0.203 | 0.751 ± 0.001 | 0.173 ± 0.292 | 0.998 ± 0.004 | 0.885 ± 0.229 |
| 200 | 0.943 ± 0.114 | 0.964 ± 0.072 | 0.450 ± 0.367 | 0.650 ± 0.339 | 1.000 ± 0.000 | 0.950 ± 0.100 |

#### DoS

| N | balanced random (oracle) | random-blind | active | coreset | cluster | stratified |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.571 ± 0.211 | 0.539 ± 0.184 | 0.213 ± 0.234 | 0.560 ± 0.073 | 0.478 ± 0.172 | 0.556 ± 0.148 |
| 25 | 0.510 ± 0.298 | 0.791 ± 0.059 | 0.286 ± 0.245 | 0.570 ± 0.262 | 0.731 ± 0.366 | 0.534 ± 0.290 |
| 50 | 0.891 ± 0.083 | 0.843 ± 0.109 | 0.169 ± 0.252 | 0.688 ± 0.006 | 0.863 ± 0.092 | 0.921 ± 0.100 |
| 100 | 0.932 ± 0.062 | 0.772 ± 0.387 | 0.110 ± 0.118 | 0.737 ± 0.334 | 0.774 ± 0.170 | 0.889 ± 0.101 |
| 200 | 0.970 ± 0.031 | 0.762 ± 0.222 | 0.394 ± 0.320 | 0.913 ± 0.063 | 0.916 ± 0.078 | 0.994 ± 0.005 |

#### Bot

| N | balanced random (oracle) | random-blind | active | coreset | cluster | stratified |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.394 ± 0.367 | 0.494 ± 0.315 | 0.000 ± 0.000 | 0.291 ± 0.237 | 0.399 ± 0.372 | 0.570 ± 0.372 |
| 25 | 0.782 ± 0.391 | 0.886 ± 0.195 | 0.001 ± 0.001 | 0.098 ± 0.195 | 0.698 ± 0.396 | 0.670 ± 0.382 |
| 50 | 0.863 ± 0.192 | 0.970 ± 0.014 | 0.002 ± 0.001 | 0.197 ± 0.241 | 0.987 ± 0.008 | 0.583 ± 0.476 |
| 100 | 0.980 ± 0.015 | 0.974 ± 0.013 | 0.001 ± 0.001 | 0.496 ± 0.012 | 0.970 ± 0.030 | 0.985 ± 0.010 |
| 200 | 0.983 ± 0.007 | 0.790 ± 0.339 | 0.256 ± 0.297 | 0.503 ± 0.002 | 0.976 ± 0.029 | 0.971 ± 0.019 |

The highest mean at the two decision budgets is:

| Family | winner @ N=25 | winner @ N=50 |
|---|---|---|
| brute-force | cluster, 0.957 | random-blind, 0.992 |
| DoS | random-blind, 0.791 | stratified, 0.921 |
| Bot | random-blind, 0.886 | cluster, 0.987 |

Those point winners are not a general selection win. The pre-registered rule
requires beating random-blind by more than one pooled standard deviation at
N ≤ 50 on at least two of three families. **No strategy satisfies it.** Cluster
qualifies only on Bot at N=50; stratified qualifies only on brute-force at
N=10; active and coreset qualify nowhere. Balanced random is excluded from this
comparison because it is an oracle using labels hidden from a real operator.

The deployment result is therefore deliberately modest: **WS2 keeps
stratified as its default because it guarantees that the analyst sees the full
score spectrum, not because it generally beats blind random.** Label count
alone is not a guarantee: at N=50, deployable random-blind ranges from 0.843
(DoS) to 0.992 (brute-force), and the best strategy changes by family and
budget. Active learning remains the clearest negative — a transfer-collapsed
model's confidence is not a useful guide to informative labels.
