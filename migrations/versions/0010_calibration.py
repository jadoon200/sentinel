"""Few-shot calibration batches, sampled flows, and retraining runs

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "calibration_batches",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("strategy", sa.String(length=32), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("n_flows", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "calibration_flows",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("batch_id", sa.Integer(), nullable=False),
        sa.Column("pool_row", sa.Integer(), nullable=False),
        sa.Column("features", _JSON, nullable=False),
        sa.Column("model_score", sa.Float(), nullable=False),
        sa.Column("true_label", sa.String(length=16), nullable=False),
        sa.Column("operator_label", sa.String(length=16), nullable=True),
        sa.Column("labelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["batch_id"], ["calibration_batches.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("batch_id", "pool_row"),
    )
    op.create_index("ix_calibration_flows_batch_id", "calibration_flows", ["batch_id"])
    op.create_table(
        "calibration_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("batch_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recall_before", sa.Float(), nullable=False),
        sa.Column("recall_after", sa.Float(), nullable=False),
        sa.Column("fpr_after", sa.Float(), nullable=False),
        sa.Column("auc_after", sa.Float(), nullable=False),
        sa.Column("n_labels_used", sa.Integer(), nullable=False),
        sa.Column("operator_accuracy", sa.Float(), nullable=False),
        sa.Column("metrics", _JSON, nullable=False),
        sa.ForeignKeyConstraint(["batch_id"], ["calibration_batches.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_calibration_runs_batch_id", "calibration_runs", ["batch_id"])


def downgrade() -> None:
    op.drop_index("ix_calibration_runs_batch_id", table_name="calibration_runs")
    op.drop_table("calibration_runs")
    op.drop_index("ix_calibration_flows_batch_id", table_name="calibration_flows")
    op.drop_table("calibration_flows")
    op.drop_table("calibration_batches")
