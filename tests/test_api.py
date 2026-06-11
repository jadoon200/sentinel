from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from sentinel.api.app import app, get_session
from sentinel.db.base import Base
from sentinel.db.models import (
    Alert,
    AttackTechnique,
    Campaign,
    CampaignReport,
    CampaignTechnique,
    ReportTechnique,
    ThreatReport,
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    with Session(engine) as session:
        session.add(AttackTechnique(technique_id="T1190", name="Exploit Public-Facing App"))
        session.add(
            ThreatReport(report_id="rss:a", source="rss", title="VPN flaw exploited in the wild")
        )
        session.add(
            ReportTechnique(
                report_id="rss:a", technique_id="T1190", score=0.5, corroborations=3, method="t"
            )
        )
        session.add(Campaign(campaign_id="camp:1", cve_ids=["CVE-2026-1111"], report_count=1))
        session.add(CampaignReport(campaign_id="camp:1", report_id="rss:a"))
        session.add(
            CampaignTechnique(
                campaign_id="camp:1",
                technique_id="T1190",
                corroborations=1,
                score=0.5,
                method="cve-component-fusion",
            )
        )
        session.add(
            Alert(
                alert_id=1,
                model="lightgbm-multiclass",
                day="Thursday",
                score=0.97,
                predicted_label="PortScan",
                true_label="PortScan",
                techniques=["T1046"],
            )
        )
        session.add(
            Alert(
                alert_id=2,
                model="lightgbm-multiclass",
                day="Friday",
                score=0.88,
                predicted_label="Web Attack - Sql Injection",
                true_label="Web Attack - Sql Injection",
                techniques=["T1190"],
            )
        )
        session.commit()

    def override_session() -> Iterator[Session]:
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = override_session
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_stats_counts_graph_entities(client: TestClient) -> None:
    stats = client.get("/stats").json()

    assert stats["threat_reports"] == 1
    assert stats["campaigns"] == 1
    assert stats["alerts"] == 2


def test_campaign_detail_joins_reports_and_techniques(client: TestClient) -> None:
    response = client.get("/campaigns/camp:1")

    assert response.status_code == 200
    body = response.json()
    assert body["cve_ids"] == ["CVE-2026-1111"]
    assert body["techniques"][0]["technique_id"] == "T1190"
    assert body["techniques"][0]["name"] == "Exploit Public-Facing App"
    assert body["reports"][0]["report_id"] == "rss:a"
    assert client.get("/campaigns/nope").status_code == 404


def test_alerts_endpoint_filters_by_model(client: TestClient) -> None:
    alerts = client.get("/alerts", params={"model": "lightgbm-multiclass"}).json()

    assert len(alerts) == 2
    assert alerts[0]["techniques"] == ["T1046"]  # highest score first
    assert client.get("/alerts", params={"model": "autoencoder"}).json() == []


def test_alert_context_fuses_with_campaigns_via_techniques(client: TestClient) -> None:
    with_match = client.get("/alerts/2/context").json()
    assert with_match["matched_campaigns"][0]["campaign_id"] == "camp:1"
    assert with_match["matched_campaigns"][0]["matched_techniques"] == ["T1190"]

    no_match = client.get("/alerts/1/context").json()
    assert no_match["matched_campaigns"] == []
    assert client.get("/alerts/999/context").status_code == 404


def test_technique_detail_counts_evidence(client: TestClient) -> None:
    body = client.get("/techniques/T1190").json()

    assert body["report_count"] == 1
    assert body["campaign_count"] == 1
    assert body["alert_count"] == 1  # alert 2 carries T1190
