"""End-to-end smoke tests for the two replay services that persist alerts.

`ids/replay.py` and `ids/waf_replay.py` are the only pipelines whose `main()`
writes to the database and whose code aligns several detectors' outputs back to
the originating flows by positional index — exactly the wiring that is easy to
get subtly wrong and that the per-detector smoke tests don't cover. These run
the real entry points against a tiny synthetic dataset and a throwaway SQLite
database, so an alignment or persistence regression fails in CI.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

import sentinel.db.base as db_base
from sentinel.config import get_settings
from sentinel.db.base import Base
from sentinel.db.models import Alert

_HOSTS = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
_DST = "192.168.10.50"


def _timestamp(day_dd: str, idx: int) -> str:
    # CIC-IDS2017's "%d/%m/%Y %I:%M:%S %p" format the loaders parse.
    hh = 1 + (idx // 3600) % 12
    mm = (idx // 60) % 60
    ss = idx % 60
    return f"{day_dd}/07/2017 {hh:02d}:{mm:02d}:{ss:02d} PM"


def _write_day(
    path: Path,
    day: str,
    day_dd: str,
    label_of: Callable[[int, int], str],
    n_per_host: int,
    seed: int,
) -> None:
    """One day CSV with enough per-host rows for the window/channel detectors."""
    rng = np.random.default_rng(seed)
    records = []
    for host_i, host in enumerate(_HOSTS):
        for j in range(n_per_host):
            idx = host_i * n_per_host + j
            label = label_of(host_i, j)
            attacker = label != "BENIGN"
            fwd_len = float(rng.integers(0, 200)) * (8 if attacker else 1)
            records.append(
                {
                    "Flow ID": f"{day}-{idx}",
                    "Src IP": host,
                    "Src Port": int(rng.integers(1024, 65000)),
                    "Dst IP": _DST,
                    # Attackers fan out across ports (drives the profile detector);
                    # benign traffic sticks to one service port.
                    "Dst Port": int(rng.integers(1, 1024)) if attacker else 80,
                    "Protocol": 6,
                    "Timestamp": _timestamp(day_dd, idx),
                    "Flow Duration": float(rng.exponential(1e5)) * (5 if attacker else 1),
                    "Total Fwd Packet": int(rng.integers(1, 40)) + (60 if attacker else 0),
                    "Total Bwd packets": int(rng.integers(0, 40)),
                    # Byte / packet-length columns the beacon detector needs.
                    "Total Length of Fwd Packet": fwd_len,
                    "Packet Length Mean": float(rng.integers(40, 1500)),
                    "Label": label,
                }
            )
    pd.DataFrame(records).to_csv(path / f"{day}-WorkingHours.csv", index=False)


@pytest.fixture
def flow_dataset(tmp_path: Path) -> Path:
    data = tmp_path / "cicids"
    data.mkdir()
    # Mon/Tue are the temporal-split training days; Tue carries one attack family
    # so the multiclass model has >=2 classes. Thursday is the held-out test day.
    _write_day(data, "Monday", "01", lambda h, j: "BENIGN", n_per_host=30, seed=1)
    _write_day(
        data,
        "Tuesday",
        "02",
        lambda h, j: "DoS Hulk" if (h == 0 and j % 2 == 0) else "BENIGN",
        n_per_host=30,
        seed=2,
    )
    _write_day(
        data,
        "Thursday",
        "03",
        lambda h, j: "PortScan" if h == 0 else ("Web Attack - XSS" if h == 1 else "BENIGN"),
        n_per_host=40,
        seed=3,
    )
    return data


@pytest.fixture
def sqlite_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """Point the global session factory at a throwaway SQLite file."""
    url = f"sqlite:///{tmp_path}/sentinel.db"
    monkeypatch.setenv("SENTINEL_DATABASE_URL", url)
    get_settings.cache_clear()
    monkeypatch.setattr(db_base, "_session_factory", None)
    engine = db_base.make_engine(url)
    Base.metadata.create_all(engine)
    yield engine


@pytest.mark.parametrize("extra", [[], ["--conformal"]], ids=["percentile", "conformal"])
def test_replay_main_writes_consistent_alerts(
    flow_dataset: Path, sqlite_db: Engine, extra: list[str]
) -> None:
    from sentinel.ids.replay import main

    counts = main(
        ["--data-dir", str(flow_dataset), "--anomaly-epochs", "1", "--max-alerts", "50", *extra]
    )

    # Every detector slot is reported (0 is fine — e.g. mlx-less CI skips sequence).
    expected = {
        "supervised_alerts",
        "anomaly_alerts",
        "sequence_alerts",
        "profile_alerts",
        "beacon_alerts",
    }
    assert expected <= counts.keys()
    assert all(isinstance(v, int) and v >= 0 for v in counts.values())

    with Session(sqlite_db) as session:
        rows = session.query(Alert).all()
    # The alerts table reflects exactly the reported counts, and every persisted
    # alert is attributed to a host and a known detector (the alignment payoff).
    assert len(rows) == sum(counts.values())
    assert all(a.source_host for a in rows)
    assert {a.model for a in rows} <= {
        "lightgbm-multiclass",
        "autoencoder",
        "sequence",
        "profile",
        "beacon",
    }


def _fake_corpora() -> dict[str, tuple[list[str], np.ndarray]]:
    benign = ["username=john", "page=2", "id=42", "search=hello", "color=blue", "sort=asc"] * 4
    sqli = ["1' OR '1'='1", "admin'--", "' UNION SELECT pw FROM users--", "' OR 1=1--"] * 4
    texts = benign + sqli
    labels = np.array([0] * len(benign) + [1] * len(sqli), dtype=np.int_)
    return {"httpparams": (texts, labels), "sqliv2": (texts, labels)}


def test_waf_replay_writes_sqli_alerts_and_coexists(
    sqlite_db: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pre-seed a flow alert; the WAF replay must only rebuild its own sqli rows.
    with Session(sqlite_db) as session:
        session.add(Alert(model="profile", score=1.0, source_host="10.0.0.9", techniques=["T1046"]))
        session.commit()

    monkeypatch.setattr("sentinel.ids.waf_replay.load_corpora", lambda *a, **k: _fake_corpora())
    from sentinel.ids.waf_replay import main

    counts = main(["--max-alerts", "20"])
    assert counts["sqli_alerts"] >= 1

    with Session(sqlite_db) as session:
        sqli = session.query(Alert).filter(Alert.model == "sqli").all()
        flow = session.query(Alert).filter(Alert.model == "profile").all()
    assert len(sqli) == counts["sqli_alerts"]
    assert all(a.techniques == ["T1190"] for a in sqli)
    assert len(flow) == 1  # the pre-seeded flow alert survived (coexistence)
