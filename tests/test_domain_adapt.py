import numpy as np
import pandas as pd

from sentinel.ids.domain_adapt import coral, feature_shift, stable_features


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
