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
    y18a = y18.to_numpy()
    # Target benign: half calibrates the threshold, half is the alignment/AE corpus.
    benign18 = x18[y18a == 0]
    cal_idx, adapt_idx = train_test_split(
        np.arange(len(benign18)), test_size=0.5, random_state=args.seed
    )
    median18 = np.nanmedian(x18.to_numpy(dtype=float), axis=0)

    def fill(frame: pd.DataFrame) -> np.ndarray:
        v = frame.to_numpy(dtype=float)
        return np.where(np.isnan(v), median18, v)

    results = []

    def run(name: str, model_scores_on_target: np.ndarray, benign_cal: np.ndarray) -> None:
        rec, fpr, auc = recall_at_fpr(model_scores_on_target, y18a, benign_cal, args.alpha)
        results.append((name, rec, fpr, auc))

    def supervised_scores(
        xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, cal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        xtr_df, xte_df, cal_df = pd.DataFrame(xtr), pd.DataFrame(xte), pd.DataFrame(cal)
        a, b, ya, yb = train_test_split(
            xtr_df, ytr, test_size=0.2, random_state=args.seed, stratify=ytr
        )
        model = train_lightgbm(a, ya, b, yb, params=DEFAULT_PARAMS)
        return np.asarray(model.predict(xte_df)), np.asarray(model.predict(cal_df))

    cal_target = fill(benign18.iloc[cal_idx])
    target_all = fill(x18)
    y17a = y17.to_numpy()

    # baseline
    s, c = supervised_scores(fill(x17), y17a, target_all, cal_target)
    run("baseline", s, c)

    # coral: align 2017 features to 2018 (using the adapt half of benign)
    aligned = coral(x17, benign18.iloc[adapt_idx])
    s, c = supervised_scores(aligned, y17a, target_all, cal_target)
    run("coral", s, c)

    # transfer-stable features only
    keep = stable_features(x17[y17a == 0], benign18.iloc[adapt_idx], keep_frac=0.6)
    cols = [x17.columns.get_loc(k) for k in keep]
    s, c = supervised_scores(fill(x17)[:, cols], y17a, target_all[:, cols], cal_target[:, cols])
    run(f"stable-feats({len(keep)})", s, c)

    # target benign-only autoencoder (no source labels at all)
    from sentinel.ids.anomaly import FlowScaler
    from sentinel.ids.backends import select_anomaly_backend

    _, train_ae, score_ae = select_anomaly_backend()
    scaler = FlowScaler().fit(benign18.iloc[adapt_idx])
    ae = train_ae(scaler.transform(benign18.iloc[adapt_idx]), epochs=5, seed=args.seed)
    ae_scores = score_ae(ae, scaler.transform(x18))
    ae_cal = score_ae(ae, scaler.transform(benign18.iloc[cal_idx]))
    run("target-AE", ae_scores, ae_cal)

    # few-shot: 2017 + N labelled 2018 flows
    for n in (50, 200, 1000):
        rng = np.random.default_rng(args.seed)
        atk = np.where(y18a == 1)[0]
        ben = np.where(y18a == 0)[0]
        take = np.concatenate(
            [rng.choice(atk, n // 2, replace=False), rng.choice(ben, n // 2, replace=False)]
        )
        xtr = np.vstack([fill(x17), target_all[take]])
        ytr = np.concatenate([y17a, y18a[take]])
        s, c = supervised_scores(xtr, ytr, target_all, cal_target)
        run(f"few-shot-{n}", s, c)

    print(f"\n{'fix':<18}{'recall':>9}{'FPR':>8}{'AUC':>8}")
    for name, rec, fpr, auc in results:
        print(f"{name:<18}{rec:>9.3f}{fpr:>8.3f}{auc:>8.3f}")


if __name__ == "__main__":
    main()
