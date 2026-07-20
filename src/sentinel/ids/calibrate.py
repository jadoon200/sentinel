"""Few-shot target-network calibration pack, sampling, and retraining.

The calibration workflow reproduces the cross-network evaluation protocol in
the product: a frozen source training set, a disjoint target labelling pool,
target-benign calibration traffic, and a held-out target test set. Operators
label only pool rows; no transform, threshold, selector, or model sees test
rows during fitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sentinel.ids.domain_adapt import select_labels
from sentinel.ids.train import DEFAULT_PARAMS, train_lightgbm

_LABEL = "__label"
_FAMILY = "__family"
_MODEL_SCORE = "__model_score"


@dataclass(frozen=True)
class EvaluationMetrics:
    recall: float
    fpr: float
    auc: float
    per_family_recall: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "recall": self.recall,
            "fpr": self.fpr,
            "auc": self.auc,
            "per_family_recall": self.per_family_recall,
        }


@dataclass(frozen=True)
class RunMetrics:
    recall_before: float
    recall_after: float
    fpr_after: float
    auc_after: float
    n_labels_used: int
    operator_accuracy: float
    per_family_recall: dict[str, float]
    baseline_per_family_recall: dict[str, float]

    def details(self) -> dict[str, Any]:
        return {
            "per_family_recall": self.per_family_recall,
            "baseline_per_family_recall": self.baseline_per_family_recall,
        }


@dataclass(frozen=True)
class CalibrationPack:
    feature_names: tuple[str, ...]
    source_x: pd.DataFrame
    source_y: NDArray[np.int64]
    pool_x: pd.DataFrame
    pool_y: NDArray[np.int64]
    pool_families: NDArray[np.str_]
    pool_scores: NDArray[np.float64]
    calibration_x: pd.DataFrame
    test_x: pd.DataFrame
    test_y: NDArray[np.int64]
    test_families: NDArray[np.str_]
    baseline: EvaluationMetrics


# Static, recorded WS3 score-stratified curve used by the UI's result chart.
# It is descriptive, not the result of the current operator's labels.
LABEL_EFFICIENCY_CURVE: list[dict[str, Any]] = [
    {
        "n": 10,
        "mean_recall": 0.708,
        "families": {"brute-force": 0.997, "DoS": 0.556, "Bot": 0.570},
    },
    {
        "n": 25,
        "mean_recall": 0.712,
        "families": {"brute-force": 0.931, "DoS": 0.534, "Bot": 0.670},
    },
    {
        "n": 50,
        "mean_recall": 0.801,
        "families": {"brute-force": 0.900, "DoS": 0.921, "Bot": 0.583},
    },
    {
        "n": 100,
        "mean_recall": 0.920,
        "families": {"brute-force": 0.885, "DoS": 0.889, "Bot": 0.985},
    },
    {
        "n": 200,
        "mean_recall": 0.972,
        "families": {"brute-force": 0.950, "DoS": 0.994, "Bot": 0.971},
    },
]


@lru_cache(maxsize=2)
def load_pack(path: Path) -> CalibrationPack:
    """Load a directory pack produced by ``build_calibration_pack.py``."""
    root = path.expanduser().resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"calibration pack metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    feature_names = tuple(str(name) for name in metadata["feature_names"])

    source = pd.read_parquet(root / "source.parquet")
    pool = pd.read_parquet(root / "pool.parquet")
    calibration = pd.read_parquet(root / "calibration.parquet")
    test = pd.read_parquet(root / "test.parquet")
    for name, frame in {
        "source": source,
        "pool": pool,
        "calibration": calibration,
        "test": test,
    }.items():
        missing = set(feature_names) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} pack frame is missing features: {sorted(missing)}")

    baseline_raw = metadata["baseline"]
    baseline = EvaluationMetrics(
        recall=float(baseline_raw["recall"]),
        fpr=float(baseline_raw["fpr"]),
        auc=float(baseline_raw["auc"]),
        per_family_recall={
            str(family): float(value) for family, value in baseline_raw["per_family_recall"].items()
        },
    )
    return CalibrationPack(
        feature_names=feature_names,
        source_x=source.loc[:, list(feature_names)],
        source_y=source[_LABEL].to_numpy(dtype=np.int64),
        pool_x=pool.loc[:, list(feature_names)],
        pool_y=pool[_LABEL].to_numpy(dtype=np.int64),
        pool_families=np.asarray(pool[_FAMILY].astype(str), dtype=np.str_),
        pool_scores=pool[_MODEL_SCORE].to_numpy(dtype=np.float64),
        calibration_x=calibration.loc[:, list(feature_names)],
        test_x=test.loc[:, list(feature_names)],
        test_y=test[_LABEL].to_numpy(dtype=np.int64),
        test_families=np.asarray(test[_FAMILY].astype(str), dtype=np.str_),
        baseline=baseline,
    )


def save_pack(pack: CalibrationPack, path: Path, *, seed: int) -> None:
    """Persist a calibration pack as four Parquet frames plus JSON metadata."""
    root = path.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    source = pack.source_x.copy()
    source[_LABEL] = pack.source_y
    pool = pack.pool_x.copy()
    pool[_LABEL] = pack.pool_y
    pool[_FAMILY] = pack.pool_families
    pool[_MODEL_SCORE] = pack.pool_scores
    calibration = pack.calibration_x.copy()
    test = pack.test_x.copy()
    test[_LABEL] = pack.test_y
    test[_FAMILY] = pack.test_families

    source.to_parquet(root / "source.parquet", index=False)
    pool.to_parquet(root / "pool.parquet", index=False)
    calibration.to_parquet(root / "calibration.parquet", index=False)
    test.to_parquet(root / "test.parquet", index=False)
    metadata = {
        "format_version": 1,
        "seed": seed,
        "feature_names": list(pack.feature_names),
        "families": sorted(set(pack.pool_families.tolist())),
        "baseline": pack.baseline.as_dict(),
        "rows": {
            "source": len(pack.source_x),
            "pool": len(pack.pool_x),
            "calibration": len(pack.calibration_x),
            "test": len(pack.test_x),
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def sample_batch(
    pack: CalibrationPack, n: int = 50, strategy: str = "stratified", seed: int = 13
) -> NDArray[np.intp]:
    """Select reproducible pool rows using the shared WS3 selector family."""
    return select_labels(
        pack.pool_x.to_numpy(dtype=np.float64),
        n,
        strategy=strategy,
        scores=pack.pool_scores,
        seed=seed,
    )


def baseline_metrics(pack: CalibrationPack) -> EvaluationMetrics:
    """Return the frozen blind-source metrics stored when the pack was built."""
    return pack.baseline


def compute_baseline_metrics(pack: CalibrationPack, *, seed: int = 13) -> EvaluationMetrics:
    """Fit the blind source model and grade it on the pack's held-out target test."""
    metrics, _ = fit_baseline(pack, seed=seed)
    return metrics


def fit_baseline(
    pack: CalibrationPack, *, seed: int = 13
) -> tuple[EvaluationMetrics, NDArray[np.float64]]:
    """Fit the blind source model, grade it, and score the selectable pool."""
    model = _fit_model(pack.source_x, pack.source_y, seed)
    metrics = _evaluate(model, pack)
    pool_scores = np.asarray(model.predict(pack.pool_x), dtype=np.float64)
    return metrics, pool_scores


def retrain(
    pack: CalibrationPack,
    labelled: list[tuple[int, int]],
    *,
    seed: int = 13,
) -> RunMetrics:
    """Retrain with operator-labelled pool rows and grade on frozen held-out test data."""
    if not labelled:
        raise ValueError("at least one operator label is required")

    # Last answer wins if a caller provides a row twice, matching the API's
    # idempotent overwrite semantics.
    answers = dict(labelled)
    rows = np.asarray(list(answers), dtype=np.intp)
    if np.any(rows < 0) or np.any(rows >= len(pack.pool_x)):
        raise IndexError("labelled pool row is outside the calibration pack")
    operator_y = np.asarray([answers[int(row)] for row in rows], dtype=np.int64)
    if not np.isin(operator_y, [0, 1]).all():
        raise ValueError("operator labels must be 0 (benign) or 1 (attack)")

    train_x = pd.concat([pack.source_x, pack.pool_x.iloc[rows]], ignore_index=True)
    train_y = np.concatenate([pack.source_y, operator_y])
    model = _fit_model(train_x, train_y, seed)
    after = _evaluate(model, pack)
    accuracy = float((operator_y == pack.pool_y[rows]).mean())
    return RunMetrics(
        recall_before=pack.baseline.recall,
        recall_after=after.recall,
        fpr_after=after.fpr,
        auc_after=after.auc,
        n_labels_used=len(rows),
        operator_accuracy=accuracy,
        per_family_recall=after.per_family_recall,
        baseline_per_family_recall=pack.baseline.per_family_recall,
    )


def _fit_model(x: pd.DataFrame, y: NDArray[np.int64], seed: int) -> Any:
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    params = {
        **DEFAULT_PARAMS,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
    }
    return train_lightgbm(x_train, y_train, x_valid, y_valid, params=params)


def _evaluate(model: Any, pack: CalibrationPack, alpha: float = 0.01) -> EvaluationMetrics:
    calibration_scores = np.asarray(model.predict(pack.calibration_x))
    scores = np.asarray(model.predict(pack.test_x))
    threshold = float(np.quantile(calibration_scores, 1.0 - alpha))
    alerts = scores > threshold
    recall = float(alerts[pack.test_y == 1].mean())
    fpr = float(alerts[pack.test_y == 0].mean())
    per_family: dict[str, float] = {}
    for family in sorted(set(pack.test_families.tolist())):
        mask = (pack.test_families == family) & (pack.test_y == 1)
        per_family[family] = float(alerts[mask].mean()) if mask.any() else 0.0
    return EvaluationMetrics(
        recall=recall,
        fpr=fpr,
        auc=float(roc_auc_score(pack.test_y, scores)),
        per_family_recall=per_family,
    )
