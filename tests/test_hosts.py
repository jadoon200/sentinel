from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sentinel.correlate.hosts import host_threats
from sentinel.db.base import Base
from sentinel.db.models import (
    Alert,
    AttackTechnique,
    Campaign,
    CampaignTechnique,
    KevEntry,
)


def _seed(session: Session) -> None:
    session.add(AttackTechnique(technique_id="T1190", name="Exploit Public-Facing App"))
    session.add(Campaign(campaign_id="camp:web", cve_ids=["CVE-2026-5027"], report_count=2))
    session.add(KevEntry(cve_id="CVE-2026-5027", vendor_project="Acme"))
    session.add(
        CampaignTechnique(
            campaign_id="camp:web",
            technique_id="T1190",
            corroborations=2,
            score=0.6,
            method="fusion",
        )
    )
    # Host A: flagged by 3 detectors, technique fuses with the KEV campaign.
    for model, score, tech in [
        ("lightgbm-multiclass", 0.95, ["T1190"]),
        ("sequence", 12.0, []),
        ("profile", 8.0, ["T1046"]),
    ]:
        session.add(
            Alert(
                model=model,
                score=score,
                techniques=tech,
                source_host="10.0.0.5",
                true_label="Web Attack",
            )
        )
    # Host B: single detector, no intel match.
    session.add(Alert(model="autoencoder", score=3.0, techniques=[], source_host="10.0.0.9"))
    # A held-out (simulated) host for the queue.
    session.add(
        Alert(
            model="lightgbm-multiclass",
            score=0.9,
            techniques=["T1190"],
            source_host="10.0.0.99",
            simulated=True,
        )
    )
    session.commit()


def test_host_rollup_ranks_and_fuses() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _seed(session)
        threats = host_threats(session, include_simulated=False)

    assert [t.host for t in threats] == ["10.0.0.5", "10.0.0.9"]  # ranked by risk
    top = threats[0]
    assert set(top.detectors) == {"lightgbm-multiclass", "sequence", "profile"}
    assert set(top.techniques) == {"T1190", "T1046"}
    assert top.fused[0].campaign_id == "camp:web"
    assert top.fused[0].kev_cves == ["CVE-2026-5027"]
    assert 0.0 < top.fused[0].fusion.strength <= 1.0  # scored, not a flat overlap
    # One drillable detection per detector (the /alerts/{id}/context entry points).
    assert {a.model for a in top.alerts} == set(top.detectors)
    assert all(a.alert_id is not None for a in top.alerts)
    assert top.risk > threats[1].risk
    assert threats[1].fused == []  # host B has no intel match


def test_simulated_hosts_are_held_out() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _seed(session)
        live = host_threats(session, include_simulated=False)
        all_hosts = host_threats(session, include_simulated=True)

    assert "10.0.0.99" not in {t.host for t in live}
    assert "10.0.0.99" in {t.host for t in all_hosts}


def test_sqli_excluded_from_flow_agreement_but_drives_severity() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        # Host 1: all five flow detectors agree.
        for model, score in [
            ("lightgbm-multiclass", 0.9),
            ("autoencoder", 9.0),
            ("sequence", 9.0),
            ("profile", 9.0),
            ("beacon", 9.0),
        ]:
            session.add(Alert(model=model, score=score, source_host="10.0.0.1", true_label="Bot"))
        # Host 2: only four flow detectors, plus a SQLi (WAF) detection.
        for model, score in [
            ("lightgbm-multiclass", 0.9),
            ("autoencoder", 9.0),
            ("sequence", 9.0),
            ("profile", 9.0),
            ("sqli", 0.5),
        ]:
            session.add(
                Alert(model=model, score=score, source_host="10.0.0.2", true_label="Web Attack")
            )
        # Hosts seen only by the SQLi detector, at high vs low confidence.
        session.add(Alert(model="sqli", score=0.95, source_host="10.0.0.3"))
        session.add(Alert(model="sqli", score=0.05, source_host="10.0.0.4"))
        session.commit()
        threats = {t.host: t for t in host_threats(session)}

    # SQLi must not substitute for a flow detector: 4 flow + SQLi ranks below 5 flow,
    # even though SQLi is still listed in the host's detector rollup.
    assert threats["10.0.0.2"].risk < threats["10.0.0.1"].risk
    assert "sqli" in threats["10.0.0.2"].detectors
    # A pure-SQLi host is still scored, and its calibrated probability drives risk.
    assert threats["10.0.0.3"].risk > threats["10.0.0.4"].risk
