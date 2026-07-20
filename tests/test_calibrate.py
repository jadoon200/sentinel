from dataclasses import replace

import numpy as np
import pandas as pd

from sentinel.ids.calibrate import (
    CalibrationPack,
    EvaluationMetrics,
    fit_baseline,
    load_pack,
    retrain,
    sample_batch,
    save_pack,
)


def synthetic_pack(seed: int = 13) -> CalibrationPack:
    rng = np.random.default_rng(seed)
    names = tuple(f"feature_{index}" for index in range(5))

    source_benign = rng.normal(0.0, 0.55, size=(1_200, len(names)))
    source_attack = rng.normal(3.0, 0.55, size=(400, len(names)))
    source_values = np.vstack([source_benign, source_attack])
    source_y = np.concatenate(
        [np.zeros(len(source_benign), dtype=np.int64), np.ones(len(source_attack), dtype=np.int64)]
    )

    pool_benign = rng.normal(8.0, 0.55, size=(300, len(names)))
    pool_attack = rng.normal(11.0, 0.55, size=(300, len(names)))
    pool_values = np.vstack([pool_benign, pool_attack])
    pool_y = np.concatenate(
        [np.zeros(len(pool_benign), dtype=np.int64), np.ones(len(pool_attack), dtype=np.int64)]
    )
    calibration_values = rng.normal(8.0, 0.55, size=(500, len(names)))
    test_benign = rng.normal(8.0, 0.55, size=(500, len(names)))
    test_attack = rng.normal(11.0, 0.55, size=(500, len(names)))
    test_values = np.vstack([test_benign, test_attack])
    test_y = np.concatenate(
        [np.zeros(len(test_benign), dtype=np.int64), np.ones(len(test_attack), dtype=np.int64)]
    )

    pack = CalibrationPack(
        feature_names=names,
        source_x=pd.DataFrame(source_values, columns=names),
        source_y=source_y,
        pool_x=pd.DataFrame(pool_values, columns=names),
        pool_y=pool_y,
        pool_families=np.repeat(np.asarray(["synthetic"], dtype=np.str_), len(pool_values)),
        pool_scores=np.zeros(len(pool_values), dtype=np.float64),
        calibration_x=pd.DataFrame(calibration_values, columns=names),
        test_x=pd.DataFrame(test_values, columns=names),
        test_y=test_y,
        test_families=np.repeat(np.asarray(["synthetic"], dtype=np.str_), len(test_values)),
        baseline=EvaluationMetrics(0.0, 0.0, 0.0, {}),
    )
    baseline, scores = fit_baseline(pack, seed=seed)
    return replace(pack, baseline=baseline, pool_scores=scores)


def test_pack_round_trip_and_sampling_are_reproducible(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path)) / "pack"
    pack = synthetic_pack()
    save_pack(pack, root, seed=13)
    load_pack.cache_clear()
    loaded = load_pack(root)

    assert loaded.feature_names == pack.feature_names
    assert len(loaded.pool_x) == len(pack.pool_x)
    np.testing.assert_array_equal(
        sample_batch(loaded, n=50, strategy="stratified", seed=7),
        sample_batch(loaded, n=50, strategy="stratified", seed=7),
    )


def test_accurate_labels_recover_shift_and_noise_degrades_ranking() -> None:
    pack = synthetic_pack()
    rows = np.concatenate([np.arange(25), np.arange(300, 325)])
    accurate = [(int(row), int(pack.pool_y[row])) for row in rows]
    noisy = [
        (row, 1 - label if index % 2 == 0 else label) for index, (row, label) in enumerate(accurate)
    ]
    inverted = [(row, 1 - label) for row, label in accurate]

    good = retrain(pack, accurate)
    degraded = retrain(pack, noisy)
    bad = retrain(pack, inverted)

    assert good.operator_accuracy == 1.0
    assert good.recall_after >= 0.85
    assert good.fpr_after <= 0.02
    assert degraded.operator_accuracy == 0.5
    assert bad.operator_accuracy == 0.0
    assert good.recall_after > bad.recall_after
    assert good.auc_after > degraded.auc_after > bad.auc_after
