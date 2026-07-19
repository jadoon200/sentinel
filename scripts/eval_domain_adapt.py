"""Can we beat the cross-network transfer failure? Measured attempts, 2017 -> 2018.

Each fix is evaluated the same honest way: recall at a 1% false-positive rate
calibrated on the *target* network's own benign traffic (what a defender can
actually set). The baseline is the plain 2017-trained model.

Fixes tried:
  baseline      — train 2017, score 2018
  coral         — align 2017 feature covariance to 2018 (label-free), retrain
  stable-feats  — keep only transfer-stable features (label-free), retrain
  target-AE     — benign-only autoencoder trained on 2018 benign (no source labels)
  few-shot-N    — 2017 training + N labelled 2018 flows

Usage: python scripts/eval_domain_adapt.py
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sentinel.config import get_settings
from sentinel.ids.cross_dataset import canonical_columns, load_2018_day, shared_feature_xy
from sentinel.ids.domain_adapt import coral, stable_features
from sentinel.ids.train import DEFAULT_PARAMS, train_lightgbm


def recall_at_fpr(
    scores: np.ndarray, y: np.ndarray, benign_cal: np.ndarray, alpha: float
) -> tuple[float, float, float]:
    thr = float(np.quantile(benign_cal, 1 - alpha))
    alerts = scores > thr
    rec = float(alerts[y == 1].mean())
    fpr = float(alerts[y == 0].mean())
    auc = float(roc_auc_score(y, scores))
    return rec, fpr, auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day-2018", type=str, default="Wednesday-14-02-2018")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    settings = get_settings()
    dir2017 = settings.ids_data_dir
    src = pd.read_csv(dir2017 / "Tuesday-WorkingHours.csv", low_memory=False, skipinitialspace=True)
    src.columns = src.columns.str.strip()
    src = canonical_columns(src)
    tgt = load_2018_day(dir2017.parent / "cicids2018" / f"{args.day_2018}.csv")

    x17, x18, y17, y18, _ = shared_feature_xy(src, tgt)
    y17a, y18a = y17.to_numpy(), y18.to_numpy()

    # Disjoint 2018 split: every method is graded on the held-out TEST set; the
    # POOL supplies few-shot labels, alignment/AE corpus, and threshold
    # calibration. No flow appears in both — the few-shot model never sees a
    # test flow, so its score is honest (this is what the first pass got wrong).
    pool_idx, test_idx = train_test_split(
        np.arange(len(x18)), test_size=0.6, random_state=args.seed, stratify=y18a
    )
    # NaN-fill medians from the POOL only: a whole-day median leaks test-set
    # statistics pre-split (audited 2026-07: no material effect, but zero is better).
    median18 = np.nanmedian(x18.iloc[pool_idx].to_numpy(dtype=float), axis=0)

    def fill(frame: pd.DataFrame) -> np.ndarray:
        v = frame.to_numpy(dtype=float)
        return np.where(np.isnan(v), median18, v)

    target_all = fill(x18)
    x_test, y_test = target_all[test_idx], y18a[test_idx]
    pool_benign = pool_idx[y18a[pool_idx] == 0]
    cal_benign = target_all[pool_benign[: len(pool_benign) // 2]]
    adapt_benign_df = x18.iloc[pool_benign[len(pool_benign) // 2 :]]

    results = []

    def supervised_scores(
        xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, cal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        xtr_df, xte_df, cal_df = pd.DataFrame(xtr), pd.DataFrame(xte), pd.DataFrame(cal)
        a, b, ya, yb = train_test_split(
            xtr_df, ytr, test_size=0.2, random_state=args.seed, stratify=ytr
        )
        model = train_lightgbm(a, ya, b, yb, params=DEFAULT_PARAMS)
        return np.asarray(model.predict(xte_df)), np.asarray(model.predict(cal_df))

    def run(name: str, scores_test: np.ndarray, benign_cal: np.ndarray) -> None:
        rec, fpr, auc = recall_at_fpr(scores_test, y_test, benign_cal, args.alpha)
        results.append((name, rec, fpr, auc))

    # baseline
    s, c = supervised_scores(fill(x17), y17a, x_test, cal_benign)
    run("baseline", s, c)

    # coral: align 2017 features to the pool's benign target traffic
    aligned = coral(x17, adapt_benign_df)
    s, c = supervised_scores(aligned, y17a, x_test, cal_benign)
    run("coral", s, c)

    # transfer-stable features only
    keep = stable_features(x17[y17a == 0], adapt_benign_df, keep_frac=0.6)
    cols = [x17.columns.get_loc(k) for k in keep]
    s, c = supervised_scores(fill(x17)[:, cols], y17a, x_test[:, cols], cal_benign[:, cols])
    run(f"stable-feats({len(keep)})", s, c)

    # target benign-only autoencoder (no source labels at all)
    from sentinel.ids.anomaly import FlowScaler
    from sentinel.ids.backends import select_anomaly_backend

    _, train_ae, score_ae = select_anomaly_backend()
    scaler = FlowScaler().fit(adapt_benign_df)
    ae = train_ae(scaler.transform(adapt_benign_df), epochs=5, seed=args.seed)
    run(
        "target-AE",
        score_ae(ae, scaler.transform(x18.iloc[test_idx])),
        score_ae(ae, scaler.transform(x18.iloc[pool_benign[: len(pool_benign) // 2]])),
    )

    # few-shot: 2017 + N labelled flows drawn ONLY from the pool (never test)
    rng = np.random.default_rng(args.seed)
    pool_atk = pool_idx[y18a[pool_idx] == 1]
    pool_ben = pool_idx[y18a[pool_idx] == 0]
    for n in (50, 200, 1000):
        take = np.concatenate(
            [
                rng.choice(pool_atk, n // 2, replace=False),
                rng.choice(pool_ben, n // 2, replace=False),
            ]
        )
        xtr = np.vstack([fill(x17), target_all[take]])
        ytr = np.concatenate([y17a, y18a[take]])
        s, c = supervised_scores(xtr, ytr, x_test, cal_benign)
        run(f"few-shot-{n}", s, c)

    print(f"\n{'fix':<18}{'recall':>9}{'FPR':>8}{'AUC':>8}")
    for name, rec, fpr, auc in results:
        print(f"{name:<18}{rec:>9.3f}{fpr:>8.3f}{auc:>8.3f}")


if __name__ == "__main__":
    main()
