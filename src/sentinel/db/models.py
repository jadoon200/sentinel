from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column

from sentinel.db.base import Base

# JSONB on Postgres, plain JSON elsewhere (keeps unit tests runnable on SQLite).
JsonType = JSON().with_variant(postgresql.JSONB(), "postgresql")


class Vulnerability(Base):
    """A CVE record sourced from the NVD API."""

    __tablename__ = "vulnerabilities"

    cve_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    description: Mapped[str | None] = mapped_column(Text())
    cvss_score: Mapped[float | None] = mapped_column(Float())
    cvss_severity: Mapped[str | None] = mapped_column(String(16))
    published: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_modified: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now().astimezone()
    )


class AttackTechnique(Base):
    """A MITRE ATT&CK technique or sub-technique from the enterprise STIX catalog."""

    __tablename__ = "attack_techniques"

    technique_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text())
    tactics: Mapped[list[str] | None] = mapped_column(JsonType)
    platforms: Mapped[list[str] | None] = mapped_column(JsonType)
    is_subtechnique: Mapped[bool] = mapped_column(Boolean(), default=False)
    url: Mapped[str | None] = mapped_column(String(255))
    stix_id: Mapped[str | None] = mapped_column(String(64))
    # Real-world procedure descriptions from `uses` relationships — extra
    # retrieval corpus for the technique mapper.
    procedure_examples: Mapped[list[str] | None] = mapped_column(JsonType)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now().astimezone()
    )


class ThreatReport(Base):
    """A CTI report/post from an OSINT source (OTX pulse, RSS item)."""

    __tablename__ = "threat_reports"

    report_id: Mapped[str] = mapped_column(String(255), primary_key=True)  # "<source>:<id>"
    source: Mapped[str] = mapped_column(String(32), index=True)  # filtered + drift analytics
    title: Mapped[str] = mapped_column(Text())
    summary: Mapped[str | None] = mapped_column(Text())
    url: Mapped[str | None] = mapped_column(String(2048))
    author: Mapped[str | None] = mapped_column(String(255))
    published: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    tags: Mapped[list[str] | None] = mapped_column(JsonType)
    # Technique IDs asserted by the report author (OTX pulses) — kept separate
    # from NLP-derived report_techniques edges so they can serve as gold labels.
    attack_ids: Mapped[list[str] | None] = mapped_column(JsonType)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    nlp_tagged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now().astimezone()
    )


class ReportTechnique(Base):
    """NLP-derived edge between a threat report and an ATT&CK technique."""

    __tablename__ = "report_techniques"

    report_id: Mapped[str] = mapped_column(ForeignKey("threat_reports.report_id"), primary_key=True)
    # Indexed for technique-first lookups (e.g. /techniques/{id}); the composite
    # PK only accelerates report-first access.
    technique_id: Mapped[str] = mapped_column(
        ForeignKey("attack_techniques.technique_id"), primary_key=True, index=True
    )
    score: Mapped[float] = mapped_column(Float())
    corroborations: Mapped[int] = mapped_column(Integer())
    method: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now().astimezone()
    )


class ReportCve(Base):
    """A CVE identifier mentioned in a threat report (extracted by regex)."""

    __tablename__ = "report_cves"

    report_id: Mapped[str] = mapped_column(ForeignKey("threat_reports.report_id"), primary_key=True)
    cve_id: Mapped[str] = mapped_column(String(20), primary_key=True)


class Campaign(Base):
    """A cluster of reports linked by shared CVE mentions (derived, rebuilt each run)."""

    __tablename__ = "campaigns"

    campaign_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    cve_ids: Mapped[list[str]] = mapped_column(JsonType)
    report_count: Mapped[int] = mapped_column(Integer())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now().astimezone()
    )


class CampaignReport(Base):
    """Membership edge between a campaign and a threat report."""

    __tablename__ = "campaign_reports"

    campaign_id: Mapped[str] = mapped_column(ForeignKey("campaigns.campaign_id"), primary_key=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("threat_reports.report_id"), primary_key=True)


class CampaignTechnique(Base):
    """Technique evidence fused across all reports of a campaign."""

    __tablename__ = "campaign_techniques"

    campaign_id: Mapped[str] = mapped_column(ForeignKey("campaigns.campaign_id"), primary_key=True)
    technique_id: Mapped[str] = mapped_column(
        ForeignKey("attack_techniques.technique_id"), primary_key=True, index=True
    )
    corroborations: Mapped[int] = mapped_column(Integer())
    score: Mapped[float] = mapped_column(Float())
    method: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now().astimezone()
    )


class Alert(Base):
    """An IDS detection from replaying flows through the trained models.

    Techniques come from the predicted attack family via the curated CIC →
    ATT&CK map (supervised model) or stay empty for pure anomaly alerts, so
    alerts can be correlated with campaign/report technique evidence.
    """

    __tablename__ = "alerts"

    alert_id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    # Indexed: /alerts?model= filter and the replay's delete-by-model rebuild.
    model: Mapped[str] = mapped_column(String(32), index=True)  # "lightgbm-multiclass" | ...
    day: Mapped[str | None] = mapped_column(String(16))
    score: Mapped[float] = mapped_column(Float())
    predicted_label: Mapped[str | None] = mapped_column(String(64))
    true_label: Mapped[str | None] = mapped_column(String(64))  # known in replay, null live
    techniques: Mapped[list[str] | None] = mapped_column(JsonType)
    # Source host the detection is attributed to — used only for grouping
    # alerts into per-host threats (the fusion rollup), never as a model
    # feature. Recovered from flow data at persist time. Indexed for that grouping.
    source_host: Mapped[str | None] = mapped_column(String(64), index=True)
    # Marks held-out detections reserved for the dashboard's "simulate" queue,
    # revealed on demand to mimic a live feed (real data, shown later). Indexed:
    # the host rollup filters on it.
    simulated: Mapped[bool] = mapped_column(Boolean(), default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now().astimezone()
    )


class CalibrationBatch(Base):
    """One reproducible target-network flow-labelling session."""

    __tablename__ = "calibration_batches"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now().astimezone()
    )
    strategy: Mapped[str] = mapped_column(String(32))
    seed: Mapped[int] = mapped_column(Integer())
    n_flows: Mapped[int] = mapped_column(Integer())
    status: Mapped[str] = mapped_column(String(16), default="open")
    notes: Mapped[str | None] = mapped_column(Text())


class CalibrationFlow(Base):
    """A sampled flow; truth stays server-side until the operator labels it."""

    __tablename__ = "calibration_flows"
    __table_args__ = (UniqueConstraint("batch_id", "pool_row"),)

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("calibration_batches.id", ondelete="CASCADE"), index=True
    )
    pool_row: Mapped[int] = mapped_column(Integer())
    features: Mapped[dict[str, float]] = mapped_column(JsonType)
    model_score: Mapped[float] = mapped_column(Float())
    true_label: Mapped[str] = mapped_column(String(16))
    operator_label: Mapped[str | None] = mapped_column(String(16))
    labelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CalibrationRun(Base):
    """Before/after metrics from retraining one labelled calibration batch."""

    __tablename__ = "calibration_runs"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("calibration_batches.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now().astimezone()
    )
    recall_before: Mapped[float] = mapped_column(Float())
    recall_after: Mapped[float] = mapped_column(Float())
    fpr_after: Mapped[float] = mapped_column(Float())
    auc_after: Mapped[float] = mapped_column(Float())
    n_labels_used: Mapped[int] = mapped_column(Integer())
    operator_accuracy: Mapped[float] = mapped_column(Float())
    metrics: Mapped[dict[str, Any]] = mapped_column(JsonType)


class KevEntry(Base):
    """A CISA Known Exploited Vulnerabilities catalog entry."""

    __tablename__ = "kev_entries"

    cve_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    vendor_project: Mapped[str | None] = mapped_column(String(255))
    product: Mapped[str | None] = mapped_column(String(255))
    vulnerability_name: Mapped[str | None] = mapped_column(Text())
    short_description: Mapped[str | None] = mapped_column(Text())
    known_ransomware_use: Mapped[str | None] = mapped_column(String(32))
    date_added: Mapped[date | None] = mapped_column(Date())
    due_date: Mapped[date | None] = mapped_column(Date())
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now().astimezone()
    )
