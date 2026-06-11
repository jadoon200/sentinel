from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from sentinel.db.base import Base
from sentinel.db.models import AttackTechnique, ReportTechnique, ThreatReport
from sentinel.nlp.mapper import TechniqueMapper, load_technique_docs
from sentinel.nlp.tagging import split_sentences, tag_untagged_reports

VOCAB = ["powershell", "phishing", "registry"]

TECHNIQUES = [
    AttackTechnique(technique_id="T1059.001", name="PowerShell", description="powershell abuse"),
    AttackTechnique(technique_id="T1566", name="Phishing", description="phishing emails"),
    AttackTechnique(technique_id="T1112", name="Modify Registry", description="registry edits"),
]


class KeywordEncoder:
    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]:
        return np.asarray([[float(t.lower().count(w)) for w in VOCAB] for t in texts])


def test_split_sentences_filters_short_fragments() -> None:
    text = (
        "Short one. The actor used powershell to stage payloads! "
        "Tiny. What data was exfiltrated overnight?"
    )
    assert split_sentences(text) == [
        "The actor used powershell to stage payloads!",
        "What data was exfiltrated overnight?",
    ]


def test_tag_untagged_reports_persists_corroborated_edges() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        for technique in TECHNIQUES:
            session.add(technique)
        session.add(
            ThreatReport(
                report_id="rss:abc",
                source="rss",
                title="Campaign report mentions powershell loaders everywhere",
                summary=(
                    "The intrusion began with phishing emails sent to staff. "
                    "Operators then executed powershell scripts on every host. "
                    "Persistence used powershell profiles across the fleet."
                ),
            )
        )
        session.add(
            ThreatReport(report_id="rss:empty", source="rss", title="Too short", summary=None)
        )
        session.commit()

        mapper = TechniqueMapper(load_technique_docs(session), encoder=KeywordEncoder())
        edges = tag_untagged_reports(
            session, mapper, method="test", min_score=0.1, max_techniques=2
        )
        session.commit()

        assert edges > 0
        stored = session.scalars(
            select(ReportTechnique).order_by(ReportTechnique.corroborations.desc())
        ).all()
        assert stored[0].technique_id == "T1059.001"
        assert stored[0].corroborations >= 2
        assert all(e.report_id == "rss:abc" for e in stored)

        # Both reports are stamped, so a second run is a no-op.
        tagged = session.scalars(select(ThreatReport)).all()
        assert all(r.nlp_tagged_at is not None for r in tagged)
        assert tag_untagged_reports(session, mapper, method="test", min_score=0.1) == 0
