"""How few target labels does cross-network detection actually need — and does
smart selection beat random?

The headline result (docs/EVAL.md, eval_cross_family.py): a 2017-trained IDS
detects ~0 of the same attacks on the 2018 network at a usable threshold, and
~50 labelled target flows recover it to 0.95-0.99 recall. That used a single
budget (50) and random selection. This pushes on both:

  - **Label-efficiency curve:** recall @1% FPR as the budget N sweeps
    10 -> 200, multi-seed (mean +/- std), so the *minimum* viable labelling
    budget is a measured number, not a guess.
  - **Selection strategy:** compare a balanced random oracle with deployable
    blind random, uncertainty, score-stratified, k-center coreset, and cluster
    sampling. Honest either way — if geometry does not beat blind random,
    simplicity wins.

Same protocol as eval_cross_family: target-benign-calibrated 1% FPR, held-out
test split (the labelling pool and the test set are disjoint, no contamination).

Usage: python scripts/eval_label_efficiency.py
"""

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import lightgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sentinel.config import get_settings
from sentinel.ids.cross_dataset import canonical_columns, load_2018_day, shared_feature_xy
from sentinel.ids.domain_adapt import select_labels
from sentinel.ids.train import DEFAULT_PARAMS, train_lightgbm

DAYS = {
    "brute-force": "Wednesday-14-02-2018",
    "DoS": "Thursday-15-02-2018",
    "Bot": "Friday-02-03-2018",
}
BUDGETS = [10, 25, 50, 100, 200]
STRATEGIES = ("random", "random-blind", "active", "coreset", "cluster", "stratified")


def _recall_at_fpr(
    scores: np.ndarray, y: np.ndarray, benign_cal: np.ndarray, alpha: float
) -> float:
    thr = float(np.quantile(benign_cal, 1 - alpha))
    return float((scores > thr)[y == 1].mean())


def _fit(x_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> lightgbm.Booster:
    a, b, ya, yb = train_test_split(
        pd.DataFrame(x_tr), y_tr, test_size=0.2, random_state=seed, stratify=y_tr
    )
    return train_lightgbm(a, ya, b, yb, params={**DEFAULT_PARAMS, "seed": seed})


def _scores(model: lightgbm.Booster, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(pd.DataFrame(x)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
        help="selection strategies to evaluate (random is the balanced oracle)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=list(DAYS),
        default=list(DAYS),
        help="attack families to evaluate",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    strategies = list(dict.fromkeys(args.strategies))
    families = list(dict.fromkeys(args.families))

    settings = get_settings()
    d2017 = settings.ids_data_dir
    src = pd.read_csv(d2017 / "Tuesday-WorkingHours.csv", low_memory=False, skipinitialspace=True)
    src.columns = src.columns.str.strip()
    src = canonical_columns(src)
    if len(src) > 100_000:
        src = src.sample(100_000, random_state=args.seed).reset_index(drop=True)

    # family -> budget -> strategy -> list of recalls (over seeds)
    results: dict[str, dict[int, dict[str, list[float]]]] = {}
    baseline_recall: dict[str, float] = {}

    for family in families:
        day = DAYS[family]
        tgt = load_2018_day(d2017.parent / "cicids2018" / f"{day}.csv")
        is_atk = tgt["label"].str.upper() != "BENIGN"
        ben = tgt[~is_atk].sample(min((~is_atk).sum(), 40_000), random_state=args.seed)
        atk = tgt[is_atk].sample(min(is_atk.sum(), 40_000), random_state=args.seed)
        tgt = pd.concat([ben, atk]).reset_index(drop=True)
        x17, x18, y17, y18, _ = shared_feature_xy(src, tgt)
        y17a, y18a = y17.to_numpy(), y18.to_numpy()

        # Split before fitting any target-derived statistic. The pool supplies
        # imputation, calibration, and selection; test is scored only.
        pool_idx, test_idx = train_test_split(
            np.arange(len(x18)), test_size=0.6, random_state=args.seed, stratify=y18a
        )
        median18 = np.nanmedian(x18.iloc[pool_idx].to_numpy(dtype=float), axis=0)

        def fill(frame: pd.DataFrame, m: np.ndarray = median18) -> np.ndarray:
            v = frame.to_numpy(dtype=float)
            return np.where(np.isnan(v), m, v)

        src_x = fill(x17)
        target_all = fill(x18)
        x_test, y_test = target_all[test_idx], y18a[test_idx]
        pool_ben = pool_idx[y18a[pool_idx] == 0]
        pool_atk = pool_idx[y18a[pool_idx] == 1]
        cal = target_all[pool_ben[: len(pool_ben) // 2]]
        select_ben = pool_ben[len(pool_ben) // 2 :]  # benign available to label

        # Blind 2017 baseline at the target-calibrated 1% FPR (the collapse). The
        # same model scores the labelling pool for uncertainty sampling.
        base = _fit(src_x, y17a, args.seed)
        baseline_recall[family] = _recall_at_fpr(
            _scores(base, x_test), y_test, _scores(base, cal), args.alpha
        )
        pool_all = np.concatenate([pool_atk, select_ben])
        pool_x = target_all[pool_all]
        pool_scores = _scores(base, pool_x)

        # Geometry/uncertainty selections stay fixed across retraining seeds;
        # their variance therefore comes from the model refit. Stochastic blind
        # random and stratified sampling redraw for each seed.
        fixed_strategies = {"active", "coreset", "cluster"}
        fixed: dict[tuple[int, str], np.ndarray] = {}
        for n in BUDGETS:
            for strategy in strategies:
                if strategy in fixed_strategies:
                    fixed[(n, strategy)] = select_labels(
                        pool_x,
                        n,
                        strategy=strategy,
                        scores=pool_scores,
                        seed=args.seed,
                    )

        results[family] = {n: {strategy: [] for strategy in strategies} for n in BUDGETS}
        for seed_offset in range(args.seeds):
            run_seed = args.seed + seed_offset
            rng = np.random.default_rng(run_seed)
            for n in BUDGETS:
                for strategy in strategies:
                    if strategy == "random":
                        # Historical upper bound: balanced with hidden ground
                        # truth. This is not deployable and is labelled oracle.
                        n_attack = n // 2
                        take = np.concatenate(
                            [
                                rng.choice(pool_atk, n_attack, replace=False),
                                rng.choice(select_ben, n - n_attack, replace=False),
                            ]
                        )
                    else:
                        relative = fixed.get((n, strategy))
                        if relative is None:
                            relative = select_labels(
                                pool_x,
                                n,
                                strategy=strategy,
                                scores=pool_scores,
                                seed=run_seed,
                            )
                        take = pool_all[relative]
                    x_tr = np.vstack([src_x, target_all[take]])
                    y_tr = np.concatenate([y17a, y18a[take]])
                    model = _fit(x_tr, y_tr, run_seed)
                    results[family][n][strategy].append(
                        _recall_at_fpr(
                            _scores(model, x_test), y_test, _scores(model, cal), args.alpha
                        )
                    )
        print(f"[{family}] done (baseline recall {baseline_recall[family]:.3f})", flush=True)

    print(f"\nRecall @ {args.alpha:.0%} FPR, mean +/- std over {args.seeds} seeds")
    if "random" in strategies:
        print("random = balanced oracle (uses hidden labels); all other strategies are deployable")
    labels = {"random": "random(oracle)"}
    print(f"{'family':<12}{'N':>5}" + "".join(f"{labels.get(s, s):>20}" for s in strategies))
    for family in families:
        print(f"{family:<12}{'base':>5}{baseline_recall[family]:>20.3f}")
        for n in BUDGETS:
            formatted = []
            for strategy in strategies:
                measurements = results[family][n][strategy]
                formatted.append(f"{np.mean(measurements):.3f}+/-{np.std(measurements):.3f}")
            print(f"{family:<12}{n:>5}" + "".join(f"{value:>20}" for value in formatted))

        deployable = [strategy for strategy in strategies if strategy != "random"] or strategies
        for n in (25, 50):
            winner = max(
                deployable,
                key=lambda strategy: float(np.mean(results[family][n][strategy])),
            )
            mean = float(np.mean(results[family][n][winner]))
            print(f"[{family}] deployable winner @ N={n}: {winner} ({mean:.3f})")

    if "random-blind" in strategies:
        print("\nAcceptance audit (>1 pooled std over random-blind at N <= 50):")
        for strategy in strategies:
            if strategy in {"random", "random-blind"}:
                continue
            wins: list[str] = []
            for family in families:
                for n in (10, 25, 50):
                    candidate = np.asarray(results[family][n][strategy])
                    blind = np.asarray(results[family][n]["random-blind"])
                    pooled_std = float(np.sqrt((candidate.var() + blind.var()) / 2.0))
                    if float(candidate.mean() - blind.mean()) > pooled_std:
                        wins.append(f"{family}@{n}")
                        break
            print(f"{strategy}: {', '.join(wins) if wins else 'no qualifying family'}")


if __name__ == "__main__":
    main()
