"""How few target labels does cross-network detection actually need — and does
smart selection beat random?

The headline result (docs/EVAL.md, eval_cross_family.py): a 2017-trained IDS
detects ~0 of the same attacks on the 2018 network at a usable threshold, and
~50 labelled target flows recover it to 0.95-0.99 recall. That used a single
budget (50) and random selection. This pushes on both:

  - **Label-efficiency curve:** recall @1% FPR as the budget N sweeps
    10 -> 200, multi-seed (mean +/- std), so the *minimum* viable labelling
    budget is a measured number, not a guess.
  - **Active vs random:** does uncertainty sampling (label the target flows the
    blind 2017 model is least sure about) reach the same recall with fewer
    labels than random? Honest either way — if random ties it, random wins on
    simplicity.

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
from sentinel.ids.train import DEFAULT_PARAMS, train_lightgbm

DAYS = {
    "brute-force": "Wednesday-14-02-2018",
    "DoS": "Thursday-15-02-2018",
    "Bot": "Friday-02-03-2018",
}
BUDGETS = [10, 25, 50, 100, 200]


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
    args = parser.parse_args()

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

    for family, day in DAYS.items():
        tgt = load_2018_day(d2017.parent / "cicids2018" / f"{day}.csv")
        is_atk = tgt["label"].str.upper() != "BENIGN"
        ben = tgt[~is_atk].sample(min((~is_atk).sum(), 40_000), random_state=args.seed)
        atk = tgt[is_atk].sample(min(is_atk.sum(), 40_000), random_state=args.seed)
        tgt = pd.concat([ben, atk]).reset_index(drop=True)
        x17, x18, y17, y18, _ = shared_feature_xy(src, tgt)
        y17a, y18a = y17.to_numpy(), y18.to_numpy()
        median18 = np.nanmedian(x18.to_numpy(dtype=float), axis=0)

        def fill(frame: pd.DataFrame, m: np.ndarray = median18) -> np.ndarray:
            v = frame.to_numpy(dtype=float)
            return np.where(np.isnan(v), m, v)

        src_x = fill(x17)
        target_all = fill(x18)
        pool_idx, test_idx = train_test_split(
            np.arange(len(x18)), test_size=0.6, random_state=args.seed, stratify=y18a
        )
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
        uncertainty = -np.abs(_scores(base, target_all[pool_all]) - 0.5)  # 0.5 = least certain
        ranked_pool = pool_all[np.argsort(-uncertainty)]  # most uncertain first

        results[family] = {n: {"random": [], "active": []} for n in BUDGETS}
        for seed in range(args.seeds):
            rng = np.random.default_rng(args.seed + seed)
            for n in BUDGETS:
                # random: balanced N/2 attack + N/2 benign from the pool
                take_r = np.concatenate(
                    [
                        rng.choice(pool_atk, n // 2, replace=False),
                        rng.choice(select_ben, n // 2, replace=False),
                    ]
                )
                # active: the N most-uncertain pool flows (natural class mix)
                take_a = ranked_pool[:n]
                for strategy, take in (("random", take_r), ("active", take_a)):
                    x_tr = np.vstack([src_x, target_all[take]])
                    y_tr = np.concatenate([y17a, y18a[take]])
                    model = _fit(x_tr, y_tr, args.seed + seed)
                    results[family][n][strategy].append(
                        _recall_at_fpr(
                            _scores(model, x_test), y_test, _scores(model, cal), args.alpha
                        )
                    )
        print(f"[{family}] done (baseline recall {baseline_recall[family]:.3f})", flush=True)

    print(f"\nRecall @ {args.alpha:.0%} FPR, mean +/- std over {args.seeds} seeds")
    print(f"{'family':<12}{'N':>5}{'random':>16}{'active':>16}")
    for family in DAYS:
        print(f"{family:<12}{'base':>5}{baseline_recall[family]:>16.3f}")
        for n in BUDGETS:
            r = results[family][n]["random"]
            a = results[family][n]["active"]
            rs = f"{np.mean(r):.3f}+/-{np.std(r):.3f}"
            as_ = f"{np.mean(a):.3f}+/-{np.std(a):.3f}"
            print(f"{family:<12}{n:>5}{rs:>16}{as_:>16}")


if __name__ == "__main__":
    main()
