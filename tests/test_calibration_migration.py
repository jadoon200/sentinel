from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from sentinel.config import get_settings


def test_calibration_migration_up_and_down_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "migration.db"
    monkeypatch.setenv("SENTINEL_DATABASE_URL", f"sqlite:///{database}")
    get_settings.cache_clear()
    config = Config("alembic.ini")

    # Older migrations intentionally target Postgres JSONB. Stamp their known
    # state so this SQLite test isolates 0010's portable up/down behavior.
    command.stamp(config, "0009")
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database}")
    expected = {"calibration_batches", "calibration_flows", "calibration_runs"}
    assert expected <= set(inspect(engine).get_table_names())

    command.downgrade(config, "0009")
    assert not expected & set(inspect(engine).get_table_names())
    engine.dispose()
    get_settings.cache_clear()
