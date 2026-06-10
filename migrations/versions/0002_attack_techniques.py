"""ATT&CK technique catalog table

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-10

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "attack_techniques",
        sa.Column("technique_id", sa.String(length=16), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("tactics", postgresql.JSONB(), nullable=True),
        sa.Column("platforms", postgresql.JSONB(), nullable=True),
        sa.Column("is_subtechnique", sa.Boolean(), nullable=False),
        sa.Column("url", sa.String(length=255), nullable=True),
        sa.Column("stix_id", sa.String(length=64), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("attack_techniques")
