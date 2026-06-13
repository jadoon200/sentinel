from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://sentinel:sentinel@localhost:5432/sentinel"

    # NVD works without a key at a lower rate limit (5 req / 30 s).
    nvd_api_key: str | None = None
    nvd_api_url: str = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    nvd_page_size: int = 2000

    kev_url: str = (
        "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    )

    # OTX requires a free account key; ingestion skips cleanly when unset.
    otx_api_key: str | None = None
    otx_api_url: str = "https://otx.alienvault.com/api/v1/pulses/subscribed"
    otx_page_limit: int = 50
    otx_max_pages: int = 4

    rss_feeds: list[str] = [
        "https://www.cisa.gov/cybersecurity-advisories/all.xml",
        "https://feeds.feedburner.com/TheHackersNews",
        "https://www.bleepingcomputer.com/feed/",
        "https://isc.sans.edu/rssfeed.xml",
    ]

    attack_stix_url: str = (
        "https://raw.githubusercontent.com/mitre-attack/attack-stix-data"
        "/master/enterprise-attack/enterprise-attack.json"
    )

    # NLP technique mapping (free local models, HuggingFace)
    nlp_bi_encoder_model: str = "cisco-ai/SecureBERT2.0-biencoder"
    nlp_cross_encoder_model: str = "cisco-ai/SecureBERT2.0-cross_encoder"
    nlp_embedding_cache_dir: Path = Path("data/embedding_cache")
    # Report tagging: per-sentence retrieval depth, score floor (bi-encoder
    # cosine scale), and max stored techniques per report. Reranking is off by
    # default until cross-encoder scores are threshold-calibrated.
    nlp_tag_top_k: int = 5
    nlp_tag_min_score: float = 0.35
    nlp_tag_max_techniques: int = 5
    nlp_rerank_reports: bool = False

    # Fusion scoring: half-life (days) of the recency decay applied to a matched
    # campaign's age — a 30-day-old correlation scores half a fresh one. Governs
    # the recency factor in the alert↔campaign fusion strength (correlate/fusion.py).
    fusion_recency_half_life_days: float = 30.0

    # IDS training (corrected CIC-IDS2017; local MLflow file store by default,
    # point at http://localhost:5001 when the compose MLflow server is up)
    ids_data_dir: Path = Path("data/cicids2017")
    mlflow_tracking_uri: str = "file:./mlruns"

    http_timeout_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
