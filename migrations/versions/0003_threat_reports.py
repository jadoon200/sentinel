"""Threat reports and NLP technique edges

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-11

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "threat_reports",
        sa.Column("report_id", sa.String(length=255), primary_key=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("url", sa.String(length=2048), nullable=True),
        sa.Column("author", sa.String(length=255), nullable=True),
        sa.Column("published", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tags", postgresql.JSONB(), nullable=True),
        sa.Column("attack_ids", postgresql.JSONB(), nullable=True),
        sa.Column("raw", postgresql.JSONB(), nullable=True),
        sa.Column("nlp_tagged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "report_techniques",
        sa.Column(
            "report_id",
            sa.String(length=255),
            sa.ForeignKey("threat_reports.report_id"),
            primary_key=True,
        ),
        sa.Column(
            "technique_id",
            sa.String(length=16),
            sa.ForeignKey("attack_techniques.technique_id"),
            primary_key=True,
        ),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("corroborations", sa.Integer(), nullable=False),
        sa.Column("method", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("report_techniques")
    op.drop_table("threat_reports")
