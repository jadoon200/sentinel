"""Solving cross-network detection: which approach catches *unseen* families?

For each 2018 attack family on a different network, compares — at a target-
benign-calibrated 1% FPR, on a held-out test split (no contamination):

  baseline      — 2017-trained supervised model (brute-force), applied blind
  target-AE     — benign-only autoencoder trained on the target's OWN benign
                  traffic (zero attack labels, family-agnostic)
  few-shot-N    — 2017 + N labelled flows of this family from the target pool

Finding: the label-free target-AE does NOT clear a usable operating point on
any family (the volumetric-DoS hypothesis was wrong — it ranks DoS at AUC 0.84
but recall@1%FPR 0.001). Few-shot is the robust fix: 50 labelled flows recover
0.95-0.99 recall across all three families, including Bot, whose 2017 baseline
ranks *worse than chance* (AUC 0.40 -> 0.997 with 50 labels).

Usage: python scripts/eval_cross_family.py
"""

import argparse
import os

# lightgbm and torch both vendor libomp on macOS; loading torch first segfaults
# lightgbm's first Dataset construction. Import lightgbm here, before anything
# pulls in torch (the anomaly backend is imported lazily inside main()).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import lightgbm  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sentinel.config import get_settings
from sentinel.ids.cross_dataset import canonical_columns, load_2018_day, shared_feature_xy
from sentinel.ids.train import DEFAULT_PARAMS, train_lightgbm

DAYS = {
    "brute-force": "Wednesday-14-02-2018",
    "DoS": "Thursday-15-02-2018",
    "Bot": "Friday-02-03-2018",
}


def recall_at_fpr(
    scores: np.ndarray, y: np.ndarray, benign_cal: np.ndarray, alpha: float
) -> tuple[float, float, float]:
    thr = float(np.quantile(benign_cal, 1 - alpha))
    alerts = scores > thr
    return (
        float(alerts[y == 1].mean()),
        float(alerts[y == 0].mean()),
        float(roc_auc_score(y, scores)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    settings = get_settings()
    d2017 = settings.ids_data_dir
    src = pd.read_csv(d2017 / "Tuesday-WorkingHours.csv", low_memory=False, skipinitialspace=True)
    src.columns = src.columns.str.strip()
    src = canonical_columns(src)
    if len(src) > 150_000:
        src = src.sample(150_000, random_state=args.seed).reset_index(drop=True)

    # Deferred (torch loads after lightgbm — libomp ordering, see top of file).
    from sentinel.ids.anomaly import FlowScaler
    from sentinel.ids.backends import select_anomaly_backend

    _, train_ae, score_ae = select_anomaly_backend()

    rows = []
    for family, day in DAYS.items():
        tgt = load_2018_day(d2017.parent / "cicids2018" / f"{day}.csv")
        # Cap benign and attack separately so each family keeps plenty of both.
        is_atk = tgt["label"].str.upper() != "BENIGN"
        ben = tgt[~is_atk].sample(min((~is_atk).sum(), 75_000), random_state=args.seed)
        atk = tgt[is_atk].sample(min(is_atk.sum(), 75_000), random_state=args.seed)
        tgt = pd.concat([ben, atk]).reset_index(drop=True)
        print(f"[{family}] {len(tgt)} flows ({len(atk)} attack)", flush=True)
        x17, x18, y17, y18, _ = shared_feature_xy(src, tgt)
        y17a, y18a = y17.to_numpy(), y18.to_numpy()
        median18 = np.nanmedian(x18.to_numpy(dtype=float), axis=0)

        def fill(frame: pd.DataFrame, m: np.ndarray = median18) -> np.ndarray:
            v = frame.to_numpy(dtype=float)
            return np.where(np.isnan(v), m, v)

        pool_idx, test_idx = train_test_split(
            np.arange(len(x18)), test_size=0.6, random_state=args.seed, stratify=y18a
        )
        target_all = fill(x18)
        x_test, y_test = target_all[test_idx], y18a[test_idx]
        pool_ben = pool_idx[y18a[pool_idx] == 0]
        cal = target_all[pool_ben[: len(pool_ben) // 2]]
        ae_train = x18.iloc[pool_ben[len(pool_ben) // 2 :]]

        def add(
            name: str,
            scores: np.ndarray,
            benign_cal: np.ndarray,
            *,
            fam: str = family,
            yt: np.ndarray = y_test,
            alpha: float = args.alpha,
        ) -> None:
            rec, fpr, auc = recall_at_fpr(scores, yt, benign_cal, alpha)
            rows.append((fam, name, rec, fpr, auc))

        # baseline: 2017 supervised model, blind
        a, b, ya, yb = train_test_split(
            pd.DataFrame(fill(x17)), y17a, test_size=0.2, random_state=args.seed, stratify=y17a
        )
        model = train_lightgbm(a, ya, b, yb, params=DEFAULT_PARAMS)
        add(
            "baseline",
            np.asarray(model.predict(pd.DataFrame(x_test))),
            np.asarray(model.predict(pd.DataFrame(cal))),
        )
        print(f"[{family}] baseline done", flush=True)

        # target-AE: benign-only, family-agnostic, no source labels
        scaler = FlowScaler().fit(ae_train)
        ae = train_ae(scaler.transform(ae_train), epochs=5, seed=args.seed)
        add(
            "target-AE",
            score_ae(ae, scaler.transform(x18.iloc[test_idx])),
            score_ae(ae, scaler.transform(x18.iloc[pool_ben[: len(pool_ben) // 2]])),
        )

        # few-shot: 2017 + N labelled flows of this family from the pool
        rng = np.random.default_rng(args.seed)
        p_atk = pool_idx[y18a[pool_idx] == 1]
        for n in (50, 500):
            take = np.concatenate(
                [
                    rng.choice(p_atk, n // 2, replace=False),
                    rng.choice(pool_ben, n // 2, replace=False),
                ]
            )
            xtr = np.vstack([fill(x17), target_all[take]])
            ytr = np.concatenate([y17a, y18a[take]])
            a, b, ya, yb = train_test_split(
                pd.DataFrame(xtr), ytr, test_size=0.2, random_state=args.seed, stratify=ytr
            )
            m = train_lightgbm(a, ya, b, yb, params=DEFAULT_PARAMS)
            add(
                f"few-shot-{n}",
                np.asarray(m.predict(pd.DataFrame(x_test))),
                np.asarray(m.predict(pd.DataFrame(cal))),
            )

    print(f"\n{'family':<12}{'approach':<14}{'recall':>9}{'FPR':>8}{'AUC':>8}")
    for family, name, rec, fpr, auc in rows:
        print(f"{family:<12}{name:<14}{rec:>9.3f}{fpr:>8.3f}{auc:>8.3f}")


if __name__ == "__main__":
    main()
