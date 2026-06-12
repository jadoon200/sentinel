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
    threshold = float(
        np.percentile(
            reconstruction_errors(autoencoder, scaler.transform(holdout)),
            args.anomaly_percentile,
        )
    )
    errors = reconstruction_errors(autoencoder, scaler.transform(x_test))
    # Anomaly alerts only where the supervised model saw nothing — the
    # ensemble's job is covering unseen attack families, not double-alerting.
    is_anomaly = (errors > threshold) & ~is_attack_prediction

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
    prof_threshold = float(np.percentile(prof_scorer.score(benign_stats[prof_holdout]), 99.0))
    prof_test_stats, prof_last = build_window_stats(flows.loc[x_test.index].reset_index(drop=True))
    prof_scores = prof_scorer.score(prof_test_stats)
    dominant = prof_scorer.dominant_stat(prof_test_stats)
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
            )
        )

    with session_scope() as session:
        # Alerts are a derived artifact of one replay pass — rebuild wholesale.
        session.execute(delete(Alert))
        for alert in alerts:
            session.add(alert)

    counts = {
        "supervised_alerts": sum(a.model == "lightgbm-multiclass" for a in alerts),
        "anomaly_alerts": sum(a.model == "autoencoder" for a in alerts),
        "sequence_alerts": sum(a.model == "sequence" for a in alerts),
        "profile_alerts": sum(a.model == "profile" for a in alerts),
    }
    print(counts)
    return counts


if __name__ == "__main__":
    main()
