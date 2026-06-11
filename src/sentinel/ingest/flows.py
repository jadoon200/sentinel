"""Prefect flows for scheduled ingestion.

Each flow is a plain callable too, so tests and one-off runs don't need a
Prefect server: `python -m sentinel.ingest.flows`.
"""

from prefect import flow, task

from sentinel.config import get_settings
from sentinel.db.base import session_scope
from sentinel.ingest.attack import fetch_attack_techniques
from sentinel.ingest.kev import fetch_kev_catalog
from sentinel.ingest.nvd import fetch_recent_cves
from sentinel.ingest.otx import fetch_otx_pulses
from sentinel.ingest.rss import fetch_rss_reports
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


@task(retries=2, retry_delay_seconds=60)
def ingest_otx() -> int:
    if not get_settings().otx_api_key:
        log.info("otx_ingest_skipped", reason="no API key configured")
        return 0
    reports = fetch_otx_pulses()
    with session_scope() as session:
        for report in reports:
            session.merge(report)
    log.info("otx_ingest_complete", count=len(reports))
    return len(reports)


@task(retries=2, retry_delay_seconds=60)
def ingest_rss() -> int:
    reports = fetch_rss_reports()
    with session_scope() as session:
        for report in reports:
            session.merge(report)
    log.info("rss_ingest_complete", count=len(reports))
    return len(reports)


@task
def tag_reports() -> int:
    # Heavy ML imports stay local so the ingestion flow doesn't pay for them.
    from sentinel.nlp.encoders import BiEncoder, CrossEncoderScorer
    from sentinel.nlp.mapper import TechniqueMapper, load_technique_docs
    from sentinel.nlp.tagging import tag_untagged_reports

    settings = get_settings()
    method = settings.nlp_bi_encoder_model + ("+rerank" if settings.nlp_rerank_reports else "")
    with session_scope() as session:
        mapper = TechniqueMapper(
            load_technique_docs(session),
            encoder=BiEncoder(),
            reranker=CrossEncoderScorer() if settings.nlp_rerank_reports else None,
            cache_dir=settings.nlp_embedding_cache_dir,
            model_name=settings.nlp_bi_encoder_model,
        )
        edges = tag_untagged_reports(
            session,
            mapper,
            method=method,
            top_k_per_sentence=settings.nlp_tag_top_k,
            min_score=settings.nlp_tag_min_score,
            max_techniques=settings.nlp_tag_max_techniques,
        )
    log.info("nlp_tagging_complete", edges=edges)
    return edges


@flow(name="nlp-enrichment")
def nlp_enrichment_flow() -> dict[str, int]:
    return {"report_technique_edges": tag_reports()}


@flow(name="osint-ingestion")
def osint_ingestion_flow(nvd_window_days: int = 7) -> dict[str, int]:
    return {
        "nvd": ingest_nvd(nvd_window_days),
        "kev": ingest_kev(),
        "attack": ingest_attack(),
        "otx": ingest_otx(),
        "rss": ingest_rss(),
    }


if __name__ == "__main__":
    import sys

    configure_logging()
    if len(sys.argv) > 1 and sys.argv[1] == "enrich":
        print(nlp_enrichment_flow())
    else:
        print(osint_ingestion_flow())
