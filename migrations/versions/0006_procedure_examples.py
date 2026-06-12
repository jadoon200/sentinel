"""Technique procedure examples for retrieval enrichment

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-12

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "attack_techniques",
        sa.Column("procedure_examples", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("attack_techniques", "procedure_examples")
