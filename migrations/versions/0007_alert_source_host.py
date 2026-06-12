"""Alert source_host and simulated flag for host fusion

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-12

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("alerts", sa.Column("source_host", sa.String(length=64), nullable=True))
    op.add_column(
        "alerts",
        sa.Column("simulated", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("alerts", "simulated")
    op.drop_column("alerts", "source_host")
