"""Report CVE mentions and campaign correlation tables

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-11

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "report_cves",
        sa.Column(
            "report_id",
            sa.String(length=255),
            sa.ForeignKey("threat_reports.report_id"),
            primary_key=True,
        ),
        sa.Column("cve_id", sa.String(length=20), primary_key=True),
    )
    op.create_table(
        "campaigns",
        sa.Column("campaign_id", sa.String(length=32), primary_key=True),
        sa.Column("cve_ids", postgresql.JSONB(), nullable=False),
        sa.Column("report_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "campaign_reports",
        sa.Column(
            "campaign_id",
            sa.String(length=32),
            sa.ForeignKey("campaigns.campaign_id"),
            primary_key=True,
        ),
        sa.Column(
            "report_id",
            sa.String(length=255),
            sa.ForeignKey("threat_reports.report_id"),
            primary_key=True,
        ),
    )
    op.create_table(
        "campaign_techniques",
        sa.Column(
            "campaign_id",
            sa.String(length=32),
            sa.ForeignKey("campaigns.campaign_id"),
            primary_key=True,
        ),
        sa.Column(
            "technique_id",
            sa.String(length=16),
            sa.ForeignKey("attack_techniques.technique_id"),
            primary_key=True,
        ),
        sa.Column("corroborations", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("method", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("campaign_techniques")
    op.drop_table("campaign_reports")
    op.drop_table("campaigns")
    op.drop_table("report_cves")
