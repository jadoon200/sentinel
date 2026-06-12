"""Does label-free recalibration recover cross-dataset detection?

Trains the brute-force detector on CIC-IDS2017, scores CSE-CIC-IDS2018, then
compares two operating points on the 2018 stream:
  static     — threshold fixed from 2017 benign (what a deployed 2017 model uses)
  recalib    — split-conformal threshold from 2018's OWN benign scores (label-free)

If the static threshold collapses but the recalibrated one recovers recall at a
controlled FPR, that earns the claim that the conformal mechanism makes a
non-transferable model usable on a new network without labels.

Usage: python scripts/eval_conformal_cross.py [--alpha 0.01]
"""

import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from sentinel.config import get_settings
from sentinel.ids.conformal import conformal_pvalues
from sentinel.ids.cross_dataset import canonical_columns, load_2018_day, shared_feature_xy
from sentinel.ids.train import DEFAULT_PARAMS, train_lightgbm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day-2018", type=str, default="Wednesday-14-02-2018")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import pandas as pd

    settings = get_settings()
    dir2017 = settings.ids_data_dir
    train_frame = pd.read_csv(
        dir2017 / "Tuesday-WorkingHours.csv", low_memory=False, skipinitialspace=True
    )
    train_frame.columns = train_frame.columns.str.strip()
    train_frame = canonical_columns(train_frame)
    test_frame = load_2018_day(dir2017.parent / "cicids2018" / f"{args.day_2018}.csv")

    x17, x18, y17, y18, labels18 = shared_feature_xy(train_frame, test_frame)
    xt, xv, yt, yv = train_test_split(x17, y17, test_size=0.2, random_state=args.seed, stratify=y17)
    model = train_lightgbm(xt, yt, xv, yv, params=DEFAULT_PARAMS)

    # Scores: held-out 2017 benign (for the static threshold), and all of 2018.
    cal2017 = np.asarray(model.predict(xv[yv == 0]))
    scores18 = np.asarray(model.predict(x18))
    y18a = y18.to_numpy()
    benign18 = scores18[y18a == 0]
    attack_mask = y18a == 1

    def recall_fpr(alerts: np.ndarray) -> tuple[float, float]:
        return float(alerts[attack_mask].mean()), float(alerts[y18a == 0].mean())

    # Static: 2017-calibrated threshold transplanted onto 2018.
    static_thr = float(np.quantile(cal2017, 1 - args.alpha))
    s_rec, s_fpr = recall_fpr(scores18 > static_thr)

    # Recalibrated: conformal p-value against 2018's own benign scores (label-free
    # in the sense that only benign target traffic is needed — no attack labels).
    cal18 = benign18[: len(benign18) // 2]  # held-out 2018 benign for calibration
    p = conformal_pvalues(cal18, scores18)
    r_rec, r_fpr = recall_fpr(p <= args.alpha)

    n_attacks = int(attack_mask.sum())
    print(f"shared features {x17.shape[1]}, 2018 flows {len(x18)} ({n_attacks} attacks)\n")
    print(f"{'policy':<22}{'recall':>10}{'FPR':>10}")
    print(f"{'static (2017 thr)':<22}{s_rec:>10.4f}{s_fpr:>10.4f}")
    print(f"{'recalibrated (2018)':<22}{r_rec:>10.4f}{r_fpr:>10.4f}")
    for label in sorted(np.unique(labels18.to_numpy()[attack_mask])):
        mask = (labels18.to_numpy() == label) & attack_mask
        print(f"    {label:<18}{float((p <= args.alpha)[mask].mean()):>10.4f}")


if __name__ == "__main__":
    main()
