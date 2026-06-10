from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, Date, DateTime, Float, String, Text
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
