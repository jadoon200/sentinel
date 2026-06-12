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
