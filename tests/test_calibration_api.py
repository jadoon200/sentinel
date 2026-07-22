from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from sentinel.api.app import app, get_session
from sentinel.api.limits import RateLimiter
from sentinel.db.base import Base
from sentinel.ids.calibrate import CalibrationPack, EvaluationMetrics, RunMetrics


def _fake_pack() -> CalibrationPack:
    names = ("duration", "packet_rate")
    pool_x = pd.DataFrame(
        {
            "duration": np.linspace(0.1, 1.2, 12),
            "packet_rate": np.linspace(10.0, 120.0, 12),
        }
    )
    pool_y = np.asarray([0, 1] * 6, dtype=np.int64)
    empty = pd.DataFrame(columns=names, dtype=float)
    return CalibrationPack(
        feature_names=names,
        source_x=empty,
        source_y=np.empty(0, dtype=np.int64),
        pool_x=pool_x,
        pool_y=pool_y,
        pool_families=np.repeat(np.asarray(["test"], dtype=np.str_), len(pool_x)),
        pool_scores=np.linspace(0.05, 0.95, len(pool_x)),
        calibration_x=empty,
        test_x=empty,
        test_y=np.empty(0, dtype=np.int64),
        test_families=np.empty(0, dtype=np.str_),
        baseline=EvaluationMetrics(0.1, 0.01, 0.6, {"test": 0.1}),
    )


@pytest.fixture
def calibration_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from sentinel.api import app as api_app

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    def override_session() -> Iterator[Session]:
        with factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    monkeypatch.setattr(api_app._settings, "api_enable_calibration", True)
    monkeypatch.setattr(api_app, "_get_calibration_pack", _fake_pack)
    monkeypatch.setattr(
        api_app,
        "_calibration_rate_limiter",
        RateLimiter(max_requests=1_000, window_seconds=60),
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_calibration_is_404_when_feature_flag_is_off(
    calibration_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sentinel.api import app as api_app

    monkeypatch.setattr(api_app._settings, "api_enable_calibration", False)
    assert calibration_client.post("/calibration/batches", json={"n": 4}).status_code == 404
    assert calibration_client.get("/calibration/curve").status_code == 404


def test_full_label_relabel_simulate_and_retrain_workflow(
    calibration_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sentinel.ids.calibrate as calibrate

    created = calibration_client.post(
        "/calibration/batches", json={"n": 10, "strategy": "stratified", "seed": 7}
    )
    assert created.status_code == 200
    batch = created.json()
    assert batch["n_flows"] == 10
    assert batch["n_labelled"] == 0
    assert all("true_label" not in flow for flow in batch["flows"])

    first = batch["flows"][0]
    first_label = calibration_client.post(
        f"/calibration/flows/{first['id']}/label", json={"label": "attack"}
    )
    assert first_label.status_code == 200
    assert first_label.json()["true_label"] in {"benign", "attack"}
    relabelled = calibration_client.post(
        f"/calibration/flows/{first['id']}/label", json={"label": "benign"}
    ).json()
    assert relabelled["operator_label"] == "benign"

    for flow in batch["flows"][1:]:
        response = calibration_client.post(f"/calibration/flows/{flow['id']}/simulate-label")
        assert response.status_code == 200
        assert response.json()["operator_label"] == response.json()["true_label"]

    fake = RunMetrics(
        recall_before=0.1,
        recall_after=0.91,
        fpr_after=0.01,
        auc_after=0.98,
        n_labels_used=10,
        operator_accuracy=0.9,
        per_family_recall={"test": 0.91},
        baseline_per_family_recall={"test": 0.1},
    )
    monkeypatch.setattr(calibrate, "retrain", lambda pack, labelled, seed: fake)
    trained = calibration_client.post(f"/calibration/batches/{batch['id']}/retrain")
    assert trained.status_code == 200
    assert trained.json()["recall_after"] == 0.91

    refreshed = calibration_client.get(f"/calibration/batches/{batch['id']}").json()
    assert refreshed["status"] == "retrained"
    assert refreshed["n_labelled"] == 10
    assert len(refreshed["runs"]) == 1
    assert calibration_client.get("/calibration/curve").status_code == 200

    # Relabelling after a retrain must not demote the batch back to "labelled".
    relabel_after_run = calibration_client.post(
        f"/calibration/flows/{first['id']}/label", json={"label": "attack"}
    )
    assert relabel_after_run.status_code == 200
    still = calibration_client.get(f"/calibration/batches/{batch['id']}").json()
    assert still["status"] == "retrained"


def test_retrain_rejects_batch_with_zero_labels(calibration_client: TestClient) -> None:
    batch = calibration_client.post("/calibration/batches", json={"n": 4}).json()
    response = calibration_client.post(f"/calibration/batches/{batch['id']}/retrain")
    assert response.status_code == 422


def test_create_validates_batch_size(calibration_client: TestClient) -> None:
    assert calibration_client.post("/calibration/batches", json={"n": 0}).status_code == 422
    assert calibration_client.post("/calibration/batches", json={"n": 201}).status_code == 422


def test_create_rejects_negative_seed(calibration_client: TestClient) -> None:
    response = calibration_client.post("/calibration/batches", json={"n": 4, "seed": -1})
    assert response.status_code == 422


def test_corrupt_pack_is_503_not_500(
    calibration_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sentinel.api import app as api_app

    def broken_pack() -> CalibrationPack:
        raise KeyError("baseline")  # stale metadata.json missing a required key

    monkeypatch.setattr(api_app, "_get_calibration_pack", broken_pack)
    response = calibration_client.post("/calibration/batches", json={"n": 4})
    assert response.status_code == 503
