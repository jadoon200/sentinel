"""Can we beat the cross-network transfer failure? Measured attempts, 2017 -> 2018.

Each fix is evaluated the same honest way: recall at a 1% false-positive rate
calibrated on the *target* network's own benign traffic (what a defender can
actually set). The baseline is the plain 2017-trained model.

Fixes tried:
  baseline      — train 2017, score 2018
  coral         — align 2017 feature covariance to 2018 (label-free), retrain
  quantile-map  — transport 2017 into 2018 benign feature units, retrain
  quantile-space— express both networks in their own benign ECDF space
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
from sentinel.ids.domain_adapt import (
    BenignQuantileTransform,
    coral,
    quantile_map,
    stable_features,
)
from sentinel.ids.train import DEFAULT_PARAMS, train_lightgbm

Metric = tuple[float, float, float]


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
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="number of full-protocol seeds for the quantile rows (default: 3)",
    )
    parser.add_argument(
        "--ae-backend",
        choices=["auto", "mlx", "torch"],
        default="auto",
        help="autoencoder backend override for headless macOS runs",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")

    settings = get_settings()
    dir2017 = settings.ids_data_dir
    src = pd.read_csv(dir2017 / "Tuesday-WorkingHours.csv", low_memory=False, skipinitialspace=True)
    src.columns = src.columns.str.strip()
    src = canonical_columns(src)
    tgt = load_2018_day(dir2017.parent / "cicids2018" / f"{args.day_2018}.csv")

    x17, x18, y17, y18, _ = shared_feature_xy(src, tgt)
    y17a, y18a = y17.to_numpy(), y18.to_numpy()

    def target_split(
        seed: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
    ]:
        """Build a disjoint pool/test split with every fitted value pool-only."""
        # The POOL supplies alignment/AE data, labels, imputation medians, and
        # threshold calibration. TEST is transformed/scored only and never fit.
        pool_idx, test_idx = train_test_split(
            np.arange(len(x18)), test_size=0.6, random_state=seed, stratify=y18a
        )
        median18 = np.nanmedian(x18.iloc[pool_idx].to_numpy(dtype=float), axis=0)

        def fill(frame: pd.DataFrame) -> np.ndarray:
            values = frame.to_numpy(dtype=float)
            return np.where(np.isnan(values), median18, values)

        target_all = fill(x18)
        pool_benign = pool_idx[y18a[pool_idx] == 0]
        split_at = len(pool_benign) // 2
        cal_idx = pool_benign[:split_at]
        adapt_idx = pool_benign[split_at:]
        return (
            pool_idx,
            test_idx,
            target_all,
            target_all[test_idx],
            y18a[test_idx],
            x18.iloc[test_idx],
            x18.iloc[cal_idx],
            x18.iloc[adapt_idx],
            median18,
        )

    results: list[tuple[str, list[Metric]]] = []

    def supervised_scores(
        xtr: np.ndarray,
        ytr: np.ndarray,
        xte: np.ndarray,
        cal: np.ndarray,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        xtr_df, xte_df, cal_df = pd.DataFrame(xtr), pd.DataFrame(xte), pd.DataFrame(cal)
        split_seed = seed if seed is not None else args.seed
        a, b, ya, yb = train_test_split(
            xtr_df, ytr, test_size=0.2, random_state=split_seed, stratify=ytr
        )
        params = DEFAULT_PARAMS
        if seed is not None:
            params = {
                **DEFAULT_PARAMS,
                "seed": seed,
                "feature_fraction_seed": seed,
                "bagging_seed": seed,
                "data_random_seed": seed,
            }
        model = train_lightgbm(a, ya, b, yb, params=params)
        return np.asarray(model.predict(xte_df)), np.asarray(model.predict(cal_df))

    def metric(scores_test: np.ndarray, y_test: np.ndarray, benign_cal: np.ndarray) -> Metric:
        return recall_at_fpr(scores_test, y_test, benign_cal, args.alpha)

    (
        pool_idx,
        test_idx,
        target_all,
        x_test,
        y_test,
        _,
        cal_benign_df,
        adapt_benign_df,
        median18,
    ) = target_split(args.seed)
    cal_benign = target_all[cal_benign_df.index]
    pool_benign = pool_idx[y18a[pool_idx] == 0]

    def fill(frame: pd.DataFrame) -> np.ndarray:
        values = frame.to_numpy(dtype=float)
        return np.where(np.isnan(values), median18, values)

    def run(name: str, scores_test: np.ndarray, benign_cal: np.ndarray) -> None:
        results.append((name, [metric(scores_test, y_test, benign_cal)]))

    # baseline
    s, c = supervised_scores(fill(x17), y17a, x_test, cal_benign)
    run("baseline", s, c)

    # coral: align 2017 features to the pool's benign target traffic
    aligned = coral(x17, adapt_benign_df)
    s, c = supervised_scores(aligned, y17a, x_test, cal_benign)
    run("coral", s, c)

    # Quantile methods get full-protocol multi-seed evaluation: every seed
    # redraws the disjoint pool/test split and refits only on pool benign flows.
    quantile_results: dict[str, list[Metric]] = {"quantile-map": [], "quantile-space": []}
    source_benign_df = x17[y17a == 0]
    source_transform = BenignQuantileTransform().fit(source_benign_df)
    source_space = source_transform.transform(x17)
    for seed in range(args.seed, args.seed + args.seeds):
        _, _, _, _, seed_y_test, seed_test_df, seed_cal_df, seed_adapt_df, _ = target_split(seed)

        mapped_source = quantile_map(x17, source_benign_df, seed_adapt_df)
        median_seed = np.nanmedian(seed_adapt_df.to_numpy(dtype=float), axis=0)
        seed_test_values = seed_test_df.to_numpy(dtype=float)
        seed_cal_values = seed_cal_df.to_numpy(dtype=float)

        mapped_scores, mapped_cal = supervised_scores(
            mapped_source,
            y17a,
            np.where(np.isnan(seed_test_values), median_seed, seed_test_values),
            np.where(np.isnan(seed_cal_values), median_seed, seed_cal_values),
            seed=seed,
        )
        quantile_results["quantile-map"].append(metric(mapped_scores, seed_y_test, mapped_cal))

        target_transform = BenignQuantileTransform().fit(seed_adapt_df)
        space_scores, space_cal = supervised_scores(
            source_space,
            y17a,
            target_transform.transform(seed_test_df),
            target_transform.transform(seed_cal_df),
            seed=seed,
        )
        quantile_results["quantile-space"].append(metric(space_scores, seed_y_test, space_cal))

    results.extend(quantile_results.items())

    # transfer-stable features only
    keep = stable_features(x17[y17a == 0], adapt_benign_df, keep_frac=0.6)
    cols = [x17.columns.get_loc(k) for k in keep]
    s, c = supervised_scores(fill(x17)[:, cols], y17a, x_test[:, cols], cal_benign[:, cols])
    run(f"stable-feats({len(keep)})", s, c)

    # target benign-only autoencoder (no source labels at all)
    from sentinel.ids.anomaly import FlowScaler
    from sentinel.ids.backends import select_anomaly_backend

    _, train_ae, score_ae = select_anomaly_backend(args.ae_backend)
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

    print(
        f"\nExisting rows: single seed {args.seed}; quantile rows: "
        f"{args.seeds} full-protocol seeds ({args.seed}..{args.seed + args.seeds - 1}), mean±std."
    )
    print(f"{'fix':<22}{'recall':>16}{'FPR':>16}{'AUC':>16}")
    for name, measurements in results:
        values = np.asarray(measurements)
        means = values.mean(axis=0)
        if len(measurements) == 1:
            formatted = [f"{value:.3f}" for value in means]
        else:
            stds = values.std(axis=0)
            formatted = [f"{mean:.3f}±{std:.3f}" for mean, std in zip(means, stds, strict=True)]
        print(f"{name:<22}{formatted[0]:>16}{formatted[1]:>16}{formatted[2]:>16}")


if __name__ == "__main__":
    main()
