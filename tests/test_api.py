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
    KevEntry,
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
        session.add(
            Campaign(
                campaign_id="camp:1",
                cve_ids=["CVE-2026-1111", "CVE-2026-2222"],
                report_count=1,
            )
        )
        session.add(KevEntry(cve_id="CVE-2026-1111", vendor_project="ExampleCorp"))
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
    assert body["cve_ids"] == ["CVE-2026-1111", "CVE-2026-2222"]
    assert body["kev_cves"] == ["CVE-2026-1111"]  # only the KEV-listed one
    assert body["techniques"][0]["technique_id"] == "T1190"
    assert body["techniques"][0]["name"] == "Exploit Public-Facing App"
    assert body["reports"][0]["report_id"] == "rss:a"
    assert "age_days" in body  # recency exposed for ranking
    assert client.get("/campaigns/nope").status_code == 404


def test_campaign_osint_off_by_default(client: TestClient) -> None:
    # The OSINT-fusion bridge is off without SENTINEL_ARGUS_API_URL — empty, never errors.
    response = client.get("/campaigns/camp:1/osint")
    assert response.status_code == 200 and response.json() == []
    # unknown campaign still 404s before reaching the bridge
    assert client.get("/campaigns/nope/osint").status_code == 404


def test_campaigns_list_ranks_and_carries_recency(client: TestClient) -> None:
    campaigns = client.get("/campaigns").json()
    assert len(campaigns) == 1
    assert "age_days" in campaigns[0]
    assert campaigns[0]["kev_cves"] == ["CVE-2026-1111"]  # KEV-involved


def test_alerts_endpoint_filters_by_model(client: TestClient) -> None:
    alerts = client.get("/alerts", params={"model": "lightgbm-multiclass"}).json()

    assert len(alerts) == 2
    assert alerts[0]["techniques"] == ["T1046"]  # highest score first
    assert client.get("/alerts", params={"model": "autoencoder"}).json() == []


def test_alerts_pagination_offset(client: TestClient) -> None:
    first = client.get("/alerts", params={"limit": 1}).json()
    second = client.get("/alerts", params={"limit": 1, "offset": 1}).json()

    assert len(first) == len(second) == 1
    assert first[0]["alert_id"] != second[0]["alert_id"]


def test_alert_context_fuses_with_campaigns_via_techniques(client: TestClient) -> None:
    with_match = client.get("/alerts/2/context").json()
    top = with_match["matched_campaigns"][0]
    assert top["campaign_id"] == "camp:1"
    assert top["kev_cves"] == ["CVE-2026-1111"]
    assert top["matched_techniques"] == ["T1190"]
    # The match carries an explainable fusion score, not just an overlap flag.
    fusion = top["fusion"]
    assert 0.0 <= fusion["strength"] <= 1.0
    assert {"specificity", "recency", "corroboration", "age_days"} <= fusion.keys()

    no_match = client.get("/alerts/1/context").json()
    assert no_match["matched_campaigns"] == []
    assert client.get("/alerts/999/context").status_code == 404


def test_technique_detail_counts_evidence(client: TestClient) -> None:
    body = client.get("/techniques/T1190").json()

    assert body["report_count"] == 1
    assert body["campaign_count"] == 1
    assert body["alert_count"] == 1  # alert 2 carries T1190


def test_navigator_layer_export(client: TestClient) -> None:
    layer = client.get("/attack-navigator-layer").json()

    assert layer["domain"] == "enterprise-attack"
    scores = {t["techniqueID"]: t["score"] for t in layer["techniques"]}
    # T1190: one report edge + one campaign edge + one alert = 3
    assert scores["T1190"] == 3
    assert layer["gradient"]["maxValue"] == 3


def test_briefing_endpoint_renders(client: TestClient) -> None:
    text = client.get("/briefing").text

    assert "daily threat briefing" in text
    assert "active campaigns" in text


def test_trending_and_drift_endpoints(client: TestClient) -> None:
    assert client.get("/trending").status_code == 200
    drift = client.get("/feed-drift").json()
    assert "verdict" in drift and "population_stability_index" in drift


def test_map_techniques_runs_mapper_over_pasted_text(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch the heavy SecureBERT mapper with a stub so the endpoint wiring
    # (split → aggregate → technique metadata join) is tested without a model load.
    from sentinel.api import app as api_app
    from sentinel.nlp.mapper import TechniqueMatch

    class FakeMapper:
        def map_text(self, text: str, top_k: int = 5) -> list[TechniqueMatch]:
            return [TechniqueMatch("T1190", "Exploit Public-Facing App", 0.71)]

    monkeypatch.setattr(api_app, "_get_mapper", lambda session: FakeMapper())
    body = client.post(
        "/map-techniques",
        json={"text": "An attacker exploited the public VPN portal to gain initial access."},
    ).json()

    assert body[0]["technique_id"] == "T1190"
    assert body[0]["name"] == "Exploit Public-Facing App"  # enriched from the graph
    assert 0.0 <= body[0]["score"] <= 1.0
    assert body[0]["corroborations"] == 1


def test_map_techniques_empty_text_returns_empty(client: TestClient) -> None:
    # No sentence clears the word floor — returns early, never touching the model.
    assert client.post("/map-techniques", json={"text": ""}).json() == []


def test_map_techniques_rejects_oversized_text(client: TestClient) -> None:
    from sentinel.api import app as api_app

    oversized = "a " * (api_app._MAX_REQUEST_CHARS + 10)
    assert client.post("/map-techniques", json={"text": oversized}).status_code == 422


def test_map_techniques_rate_limited(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from sentinel.api import app as api_app
    from sentinel.api.limits import RateLimiter
    from sentinel.nlp.mapper import TechniqueMatch

    class FakeMapper:
        def map_text(self, text: str, top_k: int = 5) -> list[TechniqueMatch]:
            return [TechniqueMatch("T1190", "Exploit Public-Facing App", 0.71)]

    monkeypatch.setattr(api_app, "_get_mapper", lambda session: FakeMapper())
    monkeypatch.setattr(api_app, "_rate_limiter", RateLimiter(max_requests=1, window_seconds=60))

    text = "An attacker exploited the public VPN portal to gain initial access."
    assert client.post("/map-techniques", json={"text": text}).status_code == 200
    assert client.post("/map-techniques", json={"text": text}).status_code == 429


def test_map_techniques_sheds_load_when_mapper_saturated(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from threading import Semaphore

    from sentinel.api import app as api_app

    saturated = Semaphore(1)
    saturated.acquire()  # no slots free — the next acquire must time out
    monkeypatch.setattr(api_app, "_inference_sem", saturated)
    monkeypatch.setattr(api_app, "_INFERENCE_TIMEOUT", 0.01)

    text = "An attacker exploited the public VPN portal to gain initial access."
    assert client.post("/map-techniques", json={"text": text}).status_code == 503


def test_body_size_limit_rejects_large_payloads(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sentinel.api import app as api_app

    monkeypatch.setattr(api_app, "_MAX_BODY_BYTES", 5)  # any real body exceeds this
    resp = client.post("/map-techniques", json={"text": "hello world"})
    assert resp.status_code == 413
