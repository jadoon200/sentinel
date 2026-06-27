"""Secondary indexes on hot query columns

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-27

Indexes the columns the API and replay filter/group on but that no primary key
covers: report source/published (drift, trending, /reports), technique_id on the
edge tables (technique-first lookups), and alert model/source_host/simulated
(/alerts filter, the host rollup, and the replay's delete-by-model rebuild).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEXES: list[tuple[str, str, list[str]]] = [
    ("ix_threat_reports_source", "threat_reports", ["source"]),
    ("ix_threat_reports_published", "threat_reports", ["published"]),
    ("ix_report_techniques_technique_id", "report_techniques", ["technique_id"]),
    ("ix_campaign_techniques_technique_id", "campaign_techniques", ["technique_id"]),
    ("ix_alerts_model", "alerts", ["model"]),
    ("ix_alerts_source_host", "alerts", ["source_host"]),
    ("ix_alerts_simulated", "alerts", ["simulated"]),
]


def upgrade() -> None:
    for name, table, cols in _INDEXES:
        op.create_index(name, table, cols)


def downgrade() -> None:
    for name, table, _cols in reversed(_INDEXES):
        op.drop_index(name, table_name=table)
