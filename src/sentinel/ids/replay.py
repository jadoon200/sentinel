"""Replay corrected CIC-IDS2017 flows through the detector ensemble into alerts.

Trains on Mon-Wed, replays Thu-Fri (the temporal split), and persists the
highest-confidence detections as Alert rows tagged with ATT&CK techniques:

- a multiclass LightGBM predicts the attack *family*, which maps to techniques
  via the curated CIC -> ATT&CK map (sentinel.ids.attack_map);
- the benign-only autoencoder contributes anomaly alerts for flows the
  supervised model scores as benign (no technique tag — that's the point).

Usage (inside the sentinel conda env, Postgres up via `make up`):
    python -m sentinel.ids.replay [--sample 500000] [--max-alerts 1000]
"""

import argparse
import os
from pathlib import Path
from typing import Any

# torch and lightgbm both vendor libomp on macOS; running both in one process
# deadlocks LightGBM's OpenMP pool without this (same workaround as tests).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import lightgbm as lgb
import numpy as np
import pandas as pd

from sentinel.config import get_settings
from sentinel.ids.attack_map import techniques_for_label
from sentinel.ids.data import DAY_COLUMN, FlowScaler, load_flows, make_xy
from sentinel.ids.train import TRAIN_DAYS

MULTICLASS_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "verbose": -1,
    "seed": 13,
}


def train_multiclass(
    features: pd.DataFrame, labels: "pd.Series[str]", num_boost_round: int = 200
) -> tuple[lgb.Booster, list[str]]:
    """Train a family classifier; returns the booster and its class names."""
    classes = sorted(labels.unique())
    encoded = labels.map({name: i for i, name in enumerate(classes)})
    params = {**MULTICLASS_PARAMS, "num_class": len(classes)}
    booster = lgb.train(params, lgb.Dataset(features, label=encoded), num_boost_round)
    return booster, classes


def main(argv: list[str] | None = None) -> dict[str, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--max-alerts", type=int, default=1000, help="per model")
    parser.add_argument("--supervised-threshold", type=float, default=0.5)
    parser.add_argument("--anomaly-percentile", type=float, default=99.0)
    parser.add_argument("--anomaly-epochs", type=int, default=5)
    parser.add_argument(
        "--conformal",
        action="store_true",
        help="gate the one-sided anomaly detectors (autoencoder, profile) with "
        "the online conformal alert-budget controller instead of a fixed benign "
        "percentile (drift-robust); sequence/beacon keep the percentile",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    from sqlalchemy import delete

    from sentinel.db.base import session_scope
    from sentinel.db.models import Alert

    settings = get_settings()
    flows = load_flows(args.data_dir or settings.ids_data_dir, sample=args.sample, seed=args.seed)
    in_train = flows[DAY_COLUMN].isin(TRAIN_DAYS)
    x_train, y_train, labels_train = make_xy(flows.loc[in_train], attempted="drop")
    x_test, _, labels_test = make_xy(flows.loc[~in_train], attempted="drop")
    days_test = flows.loc[~in_train].loc[labels_test.index, DAY_COLUMN]
    # Src IP survives in the raw flows (dropped only as a model feature) — used
    # purely to attribute each detection to a host for the fusion rollup.
    hosts_test = flows.loc[~in_train].loc[labels_test.index, "Src IP"].astype(str)

    booster, classes = train_multiclass(x_train, labels_train)
    probabilities = np.asarray(booster.predict(x_test))
    predicted_idx = probabilities.argmax(axis=1)
    predicted = np.asarray(classes, dtype=object)[predicted_idx]
    confidence = probabilities.max(axis=1)
    is_attack_prediction = (predicted != "BENIGN") & (confidence >= args.supervised_threshold)

    # Backend import deferred until all LightGBM work is done; MLX preferred
    # (benchmark-backed, and immune to the macOS libomp clash with lightgbm).
    from sentinel.ids.backends import select_anomaly_backend

    backend, train_autoencoder, reconstruction_errors = select_anomaly_backend()
    print(f"anomaly backend: {backend}")

    benign_train = x_train.loc[y_train == 0]
    holdout = benign_train.sample(frac=0.1, random_state=args.seed)
    scaler = FlowScaler().fit(benign_train.drop(index=holdout.index))
    autoencoder = train_autoencoder(
        scaler.transform(benign_train.drop(index=holdout.index)),
        epochs=args.anomaly_epochs,
        seed=args.seed,
    )
    holdout_errors = reconstruction_errors(autoencoder, scaler.transform(holdout))
    errors = reconstruction_errors(autoencoder, scaler.transform(x_test))
    if args.conformal:
        # Online alert-budget control instead of a fixed benign percentile (whose
        # true FPR drifts on Thu-Fri — see conformal.py): hold the alert rate near
        # the (100 - percentile)% budget as benign traffic shifts.
        from sentinel.ids.conformal import budget_alerts

        flagged = budget_alerts(holdout_errors, errors, args.anomaly_percentile)
    else:
        threshold = float(np.percentile(holdout_errors, args.anomaly_percentile))
        flagged = errors > threshold
    # Anomaly alerts only where the supervised model saw nothing — the
    # ensemble's job is covering unseen attack families, not double-alerting.
    is_anomaly = flagged & ~is_attack_prediction

    alerts: list[Alert] = []
    supervised_order = np.argsort(-confidence)
    taken = 0
    for i in supervised_order:
        if taken >= args.max_alerts:
            break
        if not is_attack_prediction[i]:
            continue
        alerts.append(
            Alert(
                model="lightgbm-multiclass",
                day=str(days_test.iloc[i]),
                score=float(confidence[i]),
                predicted_label=str(predicted[i]),
                true_label=str(labels_test.iloc[i]),
                techniques=techniques_for_label(str(predicted[i])),
                source_host=str(hosts_test.iloc[i]),
            )
        )
        taken += 1

    anomaly_order = np.argsort(-errors)
    taken = 0
    for i in anomaly_order:
        if taken >= args.max_alerts:
            break
        if not is_anomaly[i]:
            continue
        alerts.append(
            Alert(
                model="autoencoder",
                day=str(days_test.iloc[i]),
                score=float(errors[i]),
                predicted_label=None,
                true_label=str(labels_test.iloc[i]),
                techniques=[],
                source_host=str(hosts_test.iloc[i]),
            )
        )
        taken += 1

    # Third detector: per-host sequence model (MLX-only; covers web attacks
    # the per-flow models miss — see docs/EVAL.md). Skipped where mlx is absent.
    try:
        from sentinel.ids.sequence import build_windows, train_sequence_model, window_scores
    except ImportError:
        build_windows = None  # type: ignore[assignment]
    if build_windows is not None:
        benign_seq_flows = flows.loc[benign_train.index].reset_index(drop=True)
        seq_train, _ = build_windows(benign_seq_flows, scaler.transform(benign_train))
        rng = np.random.default_rng(args.seed)
        seq_holdout = rng.random(len(seq_train)) < 0.1
        seq_model = train_sequence_model(seq_train[~seq_holdout], seed=args.seed)
        seq_threshold = float(np.percentile(window_scores(seq_model, seq_train[seq_holdout]), 99.0))
        test_seq_flows = flows.loc[x_test.index].reset_index(drop=True)
        seq_test, last_pos = build_windows(test_seq_flows, scaler.transform(x_test))
        seq_scores = window_scores(seq_model, seq_test)
        seq_alerts = np.flatnonzero(seq_scores > seq_threshold)
        for j in seq_alerts[np.argsort(-seq_scores[seq_alerts])][: args.max_alerts]:
            i = int(last_pos[j])
            alerts.append(
                Alert(
                    model="sequence",
                    day=str(days_test.iloc[i]),
                    score=float(seq_scores[j]),
                    predicted_label=None,
                    true_label=str(labels_test.iloc[i]),
                    techniques=[],
                    source_host=str(hosts_test.iloc[i]),
                )
            )

    # Fourth detector: host-profile fan-out statistics (numpy-only). The
    # dominant statistic names the alert; port fan-out maps to T1046.
    from sentinel.ids.profile import STAT_NAMES, ProfileScorer, build_window_stats

    benign_prof_flows = flows.loc[benign_train.index].reset_index(drop=True)
    benign_stats, _ = build_window_stats(benign_prof_flows)
    prof_rng = np.random.default_rng(args.seed)
    prof_holdout = prof_rng.random(len(benign_stats)) < 0.1
    prof_scorer = ProfileScorer().fit(benign_stats[~prof_holdout])
    prof_holdout_scores = prof_scorer.score(benign_stats[prof_holdout])
    prof_threshold = float(np.percentile(prof_holdout_scores, 99.0))
    prof_test_stats, prof_last = build_window_stats(flows.loc[x_test.index].reset_index(drop=True))
    prof_scores = prof_scorer.score(prof_test_stats)
    dominant = prof_scorer.dominant_stat(prof_test_stats)
    # Profile scores are one-sided (excess fan-out), so the budget controller
    # applies here too. Sequence (two-sided: also alerts suspiciously-low error)
    # and beacon (a static per-channel set, not a time-ordered stream) keep the
    # fixed percentile — the controller's rate/drift model doesn't fit them.
    if args.conformal:
        from sentinel.ids.conformal import budget_alerts

        prof_idx = np.flatnonzero(budget_alerts(prof_holdout_scores, prof_scores, 99.0))
    else:
        prof_idx = np.flatnonzero(prof_scores > prof_threshold)
    for j in prof_idx[np.argsort(-prof_scores[prof_idx])][: args.max_alerts]:
        i = int(prof_last[j])
        alerts.append(
            Alert(
                model="profile",
                day=str(days_test.iloc[i]),
                score=float(prof_scores[j]),
                predicted_label=STAT_NAMES[int(dominant[j])],
                true_label=str(labels_test.iloc[i]),
                techniques=["T1046"] if int(dominant[j]) == 0 else [],
                source_host=str(hosts_test.iloc[i]),
            )
        )

    # Fifth detector: beacon-by-data-size-dispersion (the C2 signature periodicity
    # missed — see docs/EVAL.md). Channels need the payload-less "- Attempted"
    # polls that x_test drops, so build them from the raw Thu-Fri flows; an ARES
    # C2 channel's verdict maps to T1071.001 (web-protocol C2).
    from sentinel.ids.beacon import STAT_NAMES as BEACON_STATS
    from sentinel.ids.beacon import BeaconScorer, channel_dispersion

    benign_beacon_flows = flows.loc[benign_train.index].reset_index(drop=True)
    test_full = flows.loc[~in_train].reset_index(drop=True)
    benign_ch: pd.DataFrame | None
    test_ch: pd.DataFrame | None
    try:
        benign_ch = channel_dispersion(benign_beacon_flows)
        test_ch = channel_dispersion(test_full)
    except (ValueError, KeyError):
        benign_ch = test_ch = None
    if benign_ch is not None and test_ch is not None:
        b_scorer = BeaconScorer().fit(benign_ch[BEACON_STATS].to_numpy(dtype=float))
        b_threshold = float(
            np.percentile(b_scorer.score(benign_ch[BEACON_STATS].to_numpy(dtype=float)), 99.0)
        )
        b_scores = b_scorer.score(test_ch[BEACON_STATS].to_numpy(dtype=float))
        labels_full = flows.loc[~in_train, "Label"].astype(str).str.strip().reset_index(drop=True)
        per_flow_b = pd.DataFrame(
            {
                "src": test_full["Src IP"].astype(str),
                "dst": test_full["Dst IP"].astype(str),
                "label": labels_full,
            }
        )
        attack_mode = (
            per_flow_b[per_flow_b["label"].str.upper() != "BENIGN"]
            .groupby(["src", "dst"])["label"]
            .agg(lambda x: x.mode().iloc[0])
        )
        b_idx = np.flatnonzero(b_scores > b_threshold)
        for j in b_idx[np.argsort(-b_scores[b_idx])][: args.max_alerts]:
            row = test_ch.iloc[int(j)]
            src, dst, pos = str(row["src"]), str(row["dst"]), int(row["last_pos"])
            true_label = attack_mode.get((src, dst), str(labels_full.iloc[pos]))
            alerts.append(
                Alert(
                    model="beacon",
                    day=str(test_full[DAY_COLUMN].iloc[pos]),
                    score=float(b_scores[int(j)]),
                    predicted_label="beacon-c2",
                    true_label=str(true_label),
                    techniques=techniques_for_label("Bot"),  # T1071.001
                    source_host=src,
                )
            )

    # Reserve a small held-out queue for the dashboard's "simulate" button:
    # the two attack hosts flagged by the most detectors (the richest threats).
    # Their alerts are real — just withheld from the main feed and revealed on
    # demand to mimic a live detection arriving and fusing with intel.
    by_host_detectors: dict[str, set[str]] = {}
    for a in alerts:
        if a.source_host and a.true_label and a.true_label.upper() != "BENIGN":
            by_host_detectors.setdefault(a.source_host, set()).add(a.model)
    queue_hosts = {
        host
        for host, _ in sorted(by_host_detectors.items(), key=lambda kv: len(kv[1]), reverse=True)[
            :2
        ]
    }
    for a in alerts:
        if a.source_host in queue_hosts:
            a.simulated = True

    with session_scope() as session:
        # Flow alerts are a derived artifact of one replay pass — rebuild them,
        # but leave the WAF replay's application-layer `sqli` alerts intact so the
        # two replays coexist in the alerts table.
        session.execute(delete(Alert).where(Alert.model != "sqli"))
        for alert in alerts:
            session.add(alert)

    counts = {
        "supervised_alerts": sum(a.model == "lightgbm-multiclass" for a in alerts),
        "anomaly_alerts": sum(a.model == "autoencoder" for a in alerts),
        "sequence_alerts": sum(a.model == "sequence" for a in alerts),
        "profile_alerts": sum(a.model == "profile" for a in alerts),
        "beacon_alerts": sum(a.model == "beacon" for a in alerts),
    }
    print(counts)
    return counts


if __name__ == "__main__":
    main()
