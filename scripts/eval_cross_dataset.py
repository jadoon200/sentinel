"""Cross-dataset generalization: train on CIC-IDS2017, test on CSE-CIC-IDS2018.

The honest headline number. Trains the brute-force detector on 2017 (FTP/SSH
Patator) and tests on 2018 (FTP/SSH BruteForce) — same attack, different
network/year/collector — over the intersection of canonicalized flow
features. Reports the within-2017 score on the same feature set for contrast.

Usage: python scripts/eval_cross_dataset.py [--day Wednesday-14-02-2018]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sentinel.config import get_settings
from sentinel.ids.cross_dataset import canonical_columns, load_2018_day, shared_feature_xy
from sentinel.ids.train import DEFAULT_PARAMS, train_lightgbm


def _load_2017_bruteforce(data_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        data_dir / "Tuesday-WorkingHours.csv", low_memory=False, skipinitialspace=True
    )
    frame.columns = frame.columns.str.strip()
    return canonical_columns(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-2017", type=Path, default=None)
    parser.add_argument("--day-2018", type=str, default="Wednesday-14-02-2018")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    settings = get_settings()
    dir2017 = args.data_2017 or settings.ids_data_dir
    train_frame = _load_2017_bruteforce(dir2017)
    test_frame = load_2018_day(dir2017.parent / "cicids2018" / f"{args.day_2018}.csv")

    x17, x18, y17, y18, labels18 = shared_feature_xy(train_frame, test_frame)
    print(f"shared features: {x17.shape[1]}, 2017 train rows {len(x17)}, 2018 test rows {len(x18)}")

    # Within-2017 reference on the same shared feature set.
    xa, xb, ya, yb = train_test_split(x17, y17, test_size=0.3, random_state=args.seed, stratify=y17)
    xa, xv, ya, yv = train_test_split(xa, ya, test_size=0.2, random_state=args.seed, stratify=ya)
    model = train_lightgbm(xa, ya, xv, yv, params=DEFAULT_PARAMS)
    within = roc_auc_score(yb, np.asarray(model.predict(xb)))

    # Cross-dataset: full 2017 train, evaluate on 2018.
    xt, xv, yt, yv = train_test_split(x17, y17, test_size=0.2, random_state=args.seed, stratify=y17)
    cross_model = train_lightgbm(xt, yt, xv, yv, params=DEFAULT_PARAMS)
    scores18 = np.asarray(cross_model.predict(x18))
    cross = roc_auc_score(y18, scores18)

    benign_thr = float(np.quantile(scores18[y18.to_numpy() == 0], 0.99))
    alerts = scores18 > benign_thr

    print(f"\nwithin-2017 ROC-AUC : {within:.4f}")
    print(f"cross-2018  ROC-AUC : {cross:.4f}   (drop {within - cross:+.4f})")
    print(f"cross-2018  recall@1%FPR : {float(alerts[y18 == 1].mean()):.4f}")
    for label in sorted(labels18[y18 == 1].unique()):
        mask = (labels18 == label).to_numpy()
        print(f"    recall {label}: {float(alerts[mask].mean()):.3f}")


if __name__ == "__main__":
    main()
