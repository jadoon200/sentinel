import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sentinel.ids.domain_adapt import (
    BenignQuantileTransform,
    coral,
    feature_shift,
    few_shot_training_set,
    quantile_map,
    stable_features,
)
from sentinel.ids.train import DEFAULT_PARAMS, train_lightgbm


def _lightgbm_scores(
    train_x: pd.DataFrame,
    train_y: "pd.Series[int]",
    test_x: pd.DataFrame,
    calibration_x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_x, train_y, test_size=0.2, random_state=13, stratify=train_y
    )
    model = train_lightgbm(
        x_train,
        y_train,
        x_valid,
        y_valid,
        params=DEFAULT_PARAMS,
        num_boost_round=100,
    )
    return np.asarray(model.predict(test_x)), np.asarray(model.predict(calibration_x))


def _recall_at_one_percent_fpr(
    scores: np.ndarray, labels: np.ndarray, calibration_scores: np.ndarray
) -> float:
    threshold = float(np.quantile(calibration_scores, 0.99))
    return float((scores[labels == 1] > threshold).mean())


def test_benign_quantile_space_recovers_affine_network_shift() -> None:
    rng = np.random.default_rng(13)
    source_benign = pd.DataFrame(rng.normal(0, 1, (3000, 5)))
    source_attacks = pd.DataFrame(rng.normal(4, 1, (1000, 5)))
    target_benign = pd.DataFrame(rng.normal(0, 1, (3000, 5)) * 3 + 7)
    target_attacks = pd.DataFrame(rng.normal(4, 1, (1000, 5)) * 3 + 7)

    source_x = pd.concat([source_benign, source_attacks], ignore_index=True)
    source_y = pd.Series(
        np.r_[np.zeros(len(source_benign)), np.ones(len(source_attacks))].astype(int)
    )
    target_test = pd.concat([target_benign.iloc[2000:], target_attacks], ignore_index=True)
    target_y = np.r_[np.zeros(1000), np.ones(1000)].astype(int)
    target_calibration = target_benign.iloc[1000:2000]

    baseline_scores, baseline_calibration = _lightgbm_scores(
        source_x, source_y, target_test, target_calibration
    )
    baseline_recall = _recall_at_one_percent_fpr(baseline_scores, target_y, baseline_calibration)

    source_transform = BenignQuantileTransform().fit(source_benign)
    target_transform = BenignQuantileTransform().fit(target_benign.iloc[:1000])
    quantile_scores, quantile_calibration = _lightgbm_scores(
        pd.DataFrame(source_transform.transform(source_x)),
        source_y,
        pd.DataFrame(target_transform.transform(target_test)),
        pd.DataFrame(target_transform.transform(target_calibration)),
    )
    quantile_recall = _recall_at_one_percent_fpr(quantile_scores, target_y, quantile_calibration)

    assert baseline_recall < 0.1
    assert quantile_recall > 0.9


def test_benign_quantile_transform_handles_ties_and_constants() -> None:
    benign = pd.DataFrame(
        {
            "tied": [0.0] * 9 + [10.0],
            "constant": [4.0] * 10,
            "all_nan": [np.nan] * 10,
        }
    )
    values = pd.DataFrame(
        {"tied": [0.0, 10.0], "constant": [-100.0, 100.0], "all_nan": [1.0, np.nan]}
    )

    transformed = BenignQuantileTransform().fit(benign).transform(values)

    assert transformed[0, 0] == 0.45  # midpoint of the tied ranks [0, 9)
    assert 0.0 < transformed[0, 0] < 1.0
    assert np.all(transformed[:, 1:] == 0.5)
    assert np.isfinite(transformed).all()


def test_benign_quantile_transform_is_monotone_per_feature() -> None:
    rng = np.random.default_rng(13)
    benign = pd.DataFrame(rng.normal(size=(500, 4)))
    ordered = pd.DataFrame(np.sort(rng.normal(size=(1000, 4)), axis=0))

    transformed = BenignQuantileTransform().fit(benign).transform(ordered)

    assert np.all(np.diff(transformed, axis=0) >= 0.0)


def test_quantile_map_recovers_affine_target_units() -> None:
    rng = np.random.default_rng(13)
    source_benign = pd.DataFrame(rng.normal(size=(1000, 3)))
    target_benign = source_benign * 3.0 + 7.0
    source = source_benign.iloc[[10, 200, 900]]

    mapped = quantile_map(source, source_benign, target_benign)

    assert np.allclose(mapped, source.to_numpy() * 3.0 + 7.0)


def test_coral_matches_target_covariance() -> None:
    rng = np.random.default_rng(13)
    # Source and target share structure but have different covariance/scale.
    source = pd.DataFrame(rng.normal(0, 1, (2000, 4)))
    mix = rng.normal(0, 1, (4, 4))
    target = pd.DataFrame(rng.normal(2, 1, (2000, 4)) @ mix)

    aligned = coral(source, target)

    cov_aligned = np.cov(aligned, rowvar=False)
    cov_target = np.cov(target.to_numpy(), rowvar=False)
    # After CORAL the source covariance should closely match the target's.
    assert np.abs(cov_aligned - cov_target).max() < 0.5


def test_feature_shift_and_stable_selection() -> None:
    rng = np.random.default_rng(13)
    source = pd.DataFrame({"stable": rng.normal(0, 1, 1000), "shifted": rng.normal(0, 1, 1000)})
    target = pd.DataFrame({"stable": rng.normal(0, 1, 1000), "shifted": rng.normal(8, 1, 1000)})

    shift = feature_shift(source, target)
    assert shift.index[0] == "shifted"  # the moved feature ranks first
    assert shift["shifted"] > shift["stable"]

    keep = stable_features(source, target, keep_frac=0.5)
    assert keep == ["stable"]  # only the transfer-stable feature survives


def test_few_shot_training_set_appends_balanced_target_labels() -> None:
    rng = np.random.default_rng(13)
    source_x = pd.DataFrame(rng.normal(0, 1, (200, 3)), columns=["a", "b", "c"])
    source_y = pd.Series(rng.integers(0, 2, 200))
    # Target has an extra column that must be dropped to the shared schema.
    target_x = pd.DataFrame(rng.normal(5, 1, (400, 4)), columns=["a", "b", "c", "extra"])
    target_y = pd.Series(np.r_[np.ones(200), np.zeros(200)].astype(int))

    x, y = few_shot_training_set(source_x, source_y, target_x, target_y, n_labels=40)

    assert list(x.columns) == ["a", "b", "c"]  # aligned to source schema
    assert len(x) == 240 and len(y) == 240  # 200 source + 40 target
    assert int((y.iloc[200:] == 1).sum()) == 20  # balanced: 20 attack
    assert int((y.iloc[200:] == 0).sum()) == 20  # 20 benign
