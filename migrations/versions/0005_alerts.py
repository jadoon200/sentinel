"""IDS alerts table

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-11

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "alerts",
        sa.Column("alert_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("model", sa.String(length=32), nullable=False),
        sa.Column("day", sa.String(length=16), nullable=True),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("predicted_label", sa.String(length=64), nullable=True),
        sa.Column("true_label", sa.String(length=64), nullable=True),
        sa.Column("techniques", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("alerts")
