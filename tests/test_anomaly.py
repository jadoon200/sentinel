import numpy as np
import pandas as pd

from sentinel.ids.anomaly import FlowScaler, reconstruction_errors, train_autoencoder


def test_flow_scaler_imputes_and_clips() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [10.0, 10.0, 10.0, 10.0]})
    scaler = FlowScaler().fit(train)

    scaled = scaler.transform(pd.DataFrame({"a": [np.nan, 1e9], "b": [10.0, 10.0]}))

    assert scaled[0, 0] == 0.0  # NaN imputed to median → scaled 0
    assert scaled[1, 0] == 10.0  # extreme value clipped
    assert (scaled[:, 1] == 0.0).all()  # zero-IQR column doesn't blow up


def test_autoencoder_separates_outliers() -> None:
    rng = np.random.default_rng(13)
    benign = rng.normal(0, 1, (1000, 8)).astype(np.float32)
    anomalies = rng.normal(6, 1, (100, 8)).astype(np.float32)

    model = train_autoencoder(benign, epochs=20, batch_size=256)
    benign_errors = reconstruction_errors(model, benign)
    anomaly_errors = reconstruction_errors(model, anomalies)

    threshold = np.percentile(benign_errors, 99)
    assert (anomaly_errors > threshold).mean() > 0.9
