"""Initial tables: vulnerabilities, kev_entries

Revision ID: 0001
Revises:
Create Date: 2026-06-10

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vulnerabilities",
        sa.Column("cve_id", sa.String(length=20), primary_key=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("cvss_score", sa.Float(), nullable=True),
        sa.Column("cvss_severity", sa.String(length=16), nullable=True),
        sa.Column("published", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_modified", sa.DateTime(timezone=True), nullable=True),
        sa.Column("raw", postgresql.JSONB(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "kev_entries",
        sa.Column("cve_id", sa.String(length=20), primary_key=True),
        sa.Column("vendor_project", sa.String(length=255), nullable=True),
        sa.Column("product", sa.String(length=255), nullable=True),
        sa.Column("vulnerability_name", sa.Text(), nullable=True),
        sa.Column("short_description", sa.Text(), nullable=True),
        sa.Column("known_ransomware_use", sa.String(length=32), nullable=True),
        sa.Column("date_added", sa.Date(), nullable=True),
        sa.Column("due_date", sa.Date(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("kev_entries")
    op.drop_table("vulnerabilities")
