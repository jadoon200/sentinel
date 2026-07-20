"""Build the local, release-asset-friendly few-shot calibration pack.

The pack freezes four disjoint roles: CIC-IDS2017 source training rows,
selectable CIC-IDS2018 target rows, target-benign threshold-calibration rows,
and a held-out target test. Target-derived medians are fitted on the pool only;
the test set is never used for fitting, selection, or threshold calibration.

Usage: python scripts/build_calibration_pack.py
"""

import argparse
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sentinel.config import get_settings
from sentinel.ids.calibrate import CalibrationPack, EvaluationMetrics, fit_baseline, save_pack
from sentinel.ids.cross_dataset import canonical_columns, load_2018_day, shared_feature_xy

DAYS = {
    "brute-force": "Wednesday-14-02-2018",
    "DoS": "Thursday-15-02-2018",
    "Bot": "Friday-02-03-2018",
}


def _sample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.copy()
    return frame.sample(limit, random_state=seed)


def build_pack(
    *,
    source_path: Path,
    target_dir: Path,
    source_rows: int,
    target_rows_per_class: int,
    families: tuple[str, ...],
    seed: int,
) -> CalibrationPack:
    """Build one multi-family pack without leaking held-out test statistics."""
    source = pd.read_csv(source_path, low_memory=False, skipinitialspace=True)
    source.columns = source.columns.str.strip()
    source = canonical_columns(source)
    source = _sample(source, source_rows, seed).reset_index(drop=True)

    source_x: pd.DataFrame | None = None
    source_y: np.ndarray | None = None
    feature_names: tuple[str, ...] | None = None
    pool_frames: list[pd.DataFrame] = []
    pool_labels: list[np.ndarray] = []
    pool_families: list[np.ndarray] = []
    calibration_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    test_labels: list[np.ndarray] = []
    test_families: list[np.ndarray] = []

    for offset, family in enumerate(families):
        day = DAYS[family]
        family_seed = seed + offset
        target = load_2018_day(target_dir / f"{day}.csv")
        is_attack = target["label"].str.upper() != "BENIGN"
        benign = _sample(target[~is_attack], target_rows_per_class, family_seed)
        attack = _sample(target[is_attack], target_rows_per_class, family_seed)
        target = pd.concat([benign, attack], ignore_index=True)

        x17, x18, y17, y18, _ = shared_feature_xy(source, target)
        family_features = tuple(str(name) for name in x17.columns)
        if feature_names is None:
            feature_names = family_features
        elif feature_names != family_features:
            raise ValueError(f"shared feature schema changed for {family}")

        y17_array = y17.to_numpy(dtype=np.int64)
        y18_array = y18.to_numpy(dtype=np.int64)
        pool_idx, family_test_idx = train_test_split(
            np.arange(len(x18)),
            test_size=0.6,
            random_state=seed,
            stratify=y18_array,
        )
        median = np.nanmedian(x18.iloc[pool_idx].to_numpy(dtype=np.float64), axis=0)

        def fill(
            frame: pd.DataFrame,
            family_median: np.ndarray = median,
            columns: tuple[str, ...] = family_features,
        ) -> pd.DataFrame:
            values = frame.to_numpy(dtype=np.float64)
            return pd.DataFrame(
                np.where(np.isnan(values), family_median, values),
                columns=columns,
            )

        filled_source = fill(x17)
        filled_target = fill(x18)
        if source_x is None:
            source_x = filled_source
            source_y = y17_array

        benign_pool = pool_idx[y18_array[pool_idx] == 0]
        attack_pool = pool_idx[y18_array[pool_idx] == 1]
        calibration_idx, selectable_benign_idx = np.array_split(benign_pool, 2)
        selectable_idx = np.concatenate([attack_pool, selectable_benign_idx])

        pool_frames.append(filled_target.iloc[selectable_idx].reset_index(drop=True))
        pool_labels.append(y18_array[selectable_idx])
        family_values = np.repeat(np.asarray([family], dtype=np.str_), len(selectable_idx))
        pool_families.append(family_values)
        calibration_frames.append(filled_target.iloc[calibration_idx].reset_index(drop=True))
        test_frames.append(filled_target.iloc[family_test_idx].reset_index(drop=True))
        test_labels.append(y18_array[family_test_idx])
        test_families.append(np.repeat(np.asarray([family], dtype=np.str_), len(family_test_idx)))

    if source_x is None or source_y is None or feature_names is None:
        raise ValueError("no calibration families were loaded")

    pool_x = pd.concat(pool_frames, ignore_index=True)
    pool_y = np.concatenate(pool_labels).astype(np.int64, copy=False)
    pack = CalibrationPack(
        feature_names=feature_names,
        source_x=source_x,
        source_y=source_y,
        pool_x=pool_x,
        pool_y=pool_y,
        pool_families=np.concatenate(pool_families),
        pool_scores=np.zeros(len(pool_x), dtype=np.float64),
        calibration_x=pd.concat(calibration_frames, ignore_index=True),
        test_x=pd.concat(test_frames, ignore_index=True),
        test_y=np.concatenate(test_labels).astype(np.int64, copy=False),
        test_families=np.concatenate(test_families),
        baseline=EvaluationMetrics(0.0, 0.0, 0.0, {}),
    )
    metrics, pool_scores = fit_baseline(pack, seed=seed)
    return replace(pack, baseline=metrics, pool_scores=pool_scores)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=settings.calibration_pack_path)
    parser.add_argument("--source-rows", type=int, default=100_000)
    parser.add_argument("--target-rows-per-class", type=int, default=40_000)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=list(DAYS),
        default=["DoS"],
        help="target attack families (DoS is the acceptance/demo scenario)",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.source_rows < 100 or args.target_rows_per_class < 100:
        parser.error("row limits must each be at least 100")

    data_root = settings.ids_data_dir
    pack = build_pack(
        source_path=data_root / "Tuesday-WorkingHours.csv",
        target_dir=data_root.parent / "cicids2018",
        source_rows=args.source_rows,
        target_rows_per_class=args.target_rows_per_class,
        families=tuple(dict.fromkeys(args.families)),
        seed=args.seed,
    )
    save_pack(pack, args.out, seed=args.seed)
    print(f"wrote calibration pack to {args.out}")
    print(
        f"rows: source={len(pack.source_x)}, pool={len(pack.pool_x)}, "
        f"calibration={len(pack.calibration_x)}, test={len(pack.test_x)}"
    )
    print(
        f"blind baseline: recall={pack.baseline.recall:.3f}, "
        f"FPR={pack.baseline.fpr:.3f}, AUC={pack.baseline.auc:.3f}"
    )
    for family, recall in pack.baseline.per_family_recall.items():
        print(f"  {family}: recall={recall:.3f}")


if __name__ == "__main__":
    main()
