"""Prefect flows for scheduled ingestion.

Each flow is a plain callable too, so tests and one-off runs don't need a
Prefect server: `python -m sentinel.ingest.flows`.
"""

from prefect import flow, task

from sentinel.db.base import session_scope
from sentinel.ingest.attack import fetch_attack_techniques
from sentinel.ingest.kev import fetch_kev_catalog
from sentinel.ingest.nvd import fetch_recent_cves
from sentinel.logging import configure_logging, get_logger

log = get_logger(__name__)


@task(retries=2, retry_delay_seconds=60)
def ingest_nvd(days: int = 7) -> int:
    count = 0
    with session_scope() as session:
        for vuln in fetch_recent_cves(days=days):
            session.merge(vuln)
            count += 1
    log.info("nvd_ingest_complete", count=count, window_days=days)
    return count


@task(retries=2, retry_delay_seconds=60)
def ingest_kev() -> int:
    entries = fetch_kev_catalog()
    with session_scope() as session:
        for entry in entries:
            session.merge(entry)
    log.info("kev_ingest_complete", count=len(entries))
    return len(entries)


@task(retries=2, retry_delay_seconds=60)
def ingest_attack() -> int:
    techniques = fetch_attack_techniques()
    with session_scope() as session:
        for technique in techniques:
            session.merge(technique)
    log.info("attack_ingest_complete", count=len(techniques))
    return len(techniques)


@flow(name="osint-ingestion")
def osint_ingestion_flow(nvd_window_days: int = 7) -> dict[str, int]:
    return {
        "nvd": ingest_nvd(nvd_window_days),
        "kev": ingest_kev(),
        "attack": ingest_attack(),
    }


if __name__ == "__main__":
    configure_logging()
    print(osint_ingestion_flow())
