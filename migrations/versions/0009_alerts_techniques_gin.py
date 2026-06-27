"""GIN index on alerts.techniques for JSONB containment

Revision ID: 0009
Revises: 0008
Create Date: 2026-06-27

Backs the `techniques @> '[...]'` lookup in /techniques/{id} (and any future
"alerts carrying technique X" query) so it doesn't scan every alert. Postgres
only — the column is plain JSON on other backends, which have no GIN — so this
lives in the migration rather than the model's table args (where SQLite's
create_all would choke on it).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX = "ix_alerts_techniques_gin"


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.create_index(
        _INDEX,
        "alerts",
        ["techniques"],
        postgresql_using="gin",
        postgresql_ops={"techniques": "jsonb_ops"},
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.drop_index(_INDEX, table_name="alerts")
