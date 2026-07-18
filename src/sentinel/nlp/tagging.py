"""Tag ingested threat reports with ATT&CK techniques via the TechniqueMapper.

Each report is split into sentences, every sentence is mapped to candidate
techniques, and the evidence is corroborated across sentences before the top
techniques are persisted as report_techniques edges. Reports are stamped with
nlp_tagged_at even when nothing clears the score floor, so reruns only touch
new reports.
"""

import re
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinel.db.models import ReportTechnique, ThreatReport
from sentinel.nlp.mapper import TechniqueMapper, aggregate_matches

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, min_words: int = 4, max_sentences: int = 60) -> list[str]:
    parts = (part.strip() for part in _SENTENCE_SPLIT.split(text))
    return [part for part in parts if len(part.split()) >= min_words][:max_sentences]


def tag_report(
    report: ThreatReport,
    mapper: TechniqueMapper,
    method: str,
    top_k_per_sentence: int = 5,
    min_score: float = 0.35,
    max_techniques: int = 5,
) -> list[ReportTechnique]:
    text = ". ".join(part for part in (report.title, report.summary) if part)
    sentences = split_sentences(text)
    if not sentences:
        return []
    aggregated = aggregate_matches(
        mapper.map_text(sentence, top_k=top_k_per_sentence) for sentence in sentences
    )
    # Floor first, then cap: aggregation ranks by (corroborations, score), so a
    # high-scoring single-corroboration technique can sit below floor-failing
    # entries — slicing first would spend the budget on matches the floor discards.
    qualified = [match for match in aggregated if match.score >= min_score]
    return [
        ReportTechnique(
            report_id=report.report_id,
            technique_id=match.technique_id,
            score=match.score,
            corroborations=match.corroborations,
            method=method,
        )
        for match in qualified[:max_techniques]
    ]


def untagged_reports(session: Session) -> list[ThreatReport]:
    query = select(ThreatReport).where(ThreatReport.nlp_tagged_at.is_(None))
    return list(session.scalars(query).all())


def tag_untagged_reports(
    session: Session,
    mapper: TechniqueMapper,
    method: str,
    top_k_per_sentence: int = 5,
    min_score: float = 0.35,
    max_techniques: int = 5,
) -> int:
    edges = 0
    for report in untagged_reports(session):
        for edge in tag_report(
            report,
            mapper,
            method=method,
            top_k_per_sentence=top_k_per_sentence,
            min_score=min_score,
            max_techniques=max_techniques,
        ):
            session.merge(edge)
            edges += 1
        report.nlp_tagged_at = datetime.now().astimezone()
    return edges
