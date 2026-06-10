from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from sentinel.db.base import Base
from sentinel.db.models import KevEntry, Vulnerability


def test_models_roundtrip_on_sqlite() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(
            Vulnerability(
                cve_id="CVE-2026-0001",
                description="test",
                cvss_score=7.5,
                cvss_severity="HIGH",
                raw={"id": "CVE-2026-0001"},
            )
        )
        session.add(KevEntry(cve_id="CVE-2026-0001", vendor_project="ExampleCorp"))
        session.commit()

        vuln = session.scalars(select(Vulnerability)).one()
        assert vuln.raw == {"id": "CVE-2026-0001"}
        assert session.scalars(select(KevEntry)).one().vendor_project == "ExampleCorp"
