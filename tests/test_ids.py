import numpy as np
import pandas as pd

from sentinel.ids.attack_map import techniques_for_label
from sentinel.ids.data import make_xy
from sentinel.ids.train import evaluate, train_lightgbm


def _flows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flow ID": ["a", "b", "c", "d"],
            "Src IP": ["10.0.0.1"] * 4,
            "Dst Port": [80, 22, 443, 80],
            "Flow Duration": [100.0, 200.0, np.inf, 50.0],
            "Total Fwd Packets": [10, 5, 3, 8],
            "Label": ["BENIGN", "SSH-Patator", "PortScan", "FTP-Patator - Attempted"],
        }
    )


def test_make_xy_drops_identity_columns_and_handles_inf() -> None:
    features, target, labels = make_xy(_flows(), attempted="drop")

    assert list(features.columns) == ["Flow Duration", "Total Fwd Packets"]
    assert len(features) == 3  # attempted flow dropped
    assert target.tolist() == [0, 1, 1]
    assert labels.tolist() == ["BENIGN", "SSH-Patator", "PortScan"]
    assert features["Flow Duration"].isna().sum() == 1  # inf became NaN


def test_make_xy_attempted_policies() -> None:
    _, as_benign, _ = make_xy(_flows(), attempted="benign")
    _, as_malicious, _ = make_xy(_flows(), attempted="malicious")

    assert as_benign.tolist() == [0, 1, 1, 0]
    assert as_malicious.tolist() == [0, 1, 1, 1]


def test_attack_map_normalizes_label_variants() -> None:
    assert techniques_for_label("PortScan") == ["T1046"]
    assert techniques_for_label("FTP-Patator - Attempted") == ["T1110"]
    assert techniques_for_label("Web Attack \x96 Brute Force") == ["T1110"]
    assert techniques_for_label("BENIGN") == []
    assert techniques_for_label("something new") == []


def test_train_multiclass_predicts_families_for_technique_tagging() -> None:
    from sentinel.ids.replay import train_multiclass

    rng = np.random.default_rng(13)
    n = 300
    family = pd.Series(rng.choice(["BENIGN", "PortScan", "SSH-Patator"], n)).astype(str)
    offsets = family.map({"BENIGN": 0.0, "PortScan": 4.0, "SSH-Patator": -4.0}).to_numpy()
    x = pd.DataFrame({"signal": offsets + rng.normal(0, 0.3, n), "noise": rng.normal(0, 1, n)})

    booster, classes = train_multiclass(x, family, num_boost_round=30)
    predicted_idx = np.asarray(booster.predict(x)).argmax(axis=1)
    predicted = np.asarray(classes, dtype=object)[predicted_idx]

    assert (predicted == family.to_numpy()).mean() > 0.95
    assert techniques_for_label("PortScan") == ["T1046"]


def test_train_and_evaluate_on_separable_data() -> None:
    rng = np.random.default_rng(13)
    n = 400
    # Feature 0 separates the classes; feature 1 is noise.
    y = pd.Series(rng.integers(0, 2, n)).astype(int)
    x = pd.DataFrame(
        {
            "signal": y * 2.0 + rng.normal(0, 0.2, n),
            "noise": rng.normal(0, 1.0, n),
        }
    )
    labels = pd.Series(np.where(y == 1, "PortScan", "BENIGN")).astype(str)

    model = train_lightgbm(x[:300], y[:300], x[300:], y[300:], num_boost_round=50)
    metrics = evaluate(model, x[300:], y[300:], labels[300:])

    assert metrics["roc_auc"] > 0.95
    assert metrics["recall__PortScan"] > 0.9
    assert metrics["false_positive_rate"] < 0.2
