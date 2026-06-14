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

    # Keyless CTI feeds, keyed by a short provenance label stored on each report
    # (so feed-drift and the dashboard can attribute reports to a publisher).
    # All verified live and parseable; one broken feed is skipped, not fatal.
    rss_feeds: dict[str, str] = {
        "thehackernews": "https://feeds.feedburner.com/TheHackersNews",
        "bleepingcomputer": "https://www.bleepingcomputer.com/feed/",
        "sans-isc": "https://isc.sans.edu/rssfeed.xml",
        "krebsonsecurity": "https://krebsonsecurity.com/feed/",
        "talos": "https://blog.talosintelligence.com/rss/",
        "unit42": "https://unit42.paloaltonetworks.com/feed/",
        "project-zero": "https://googleprojectzero.blogspot.com/feeds/posts/default",
        "securelist": "https://securelist.com/feed/",
        "welivesecurity": "https://www.welivesecurity.com/en/rss/feed/",
        "schneier": "https://www.schneier.com/feed/atom/",
        "checkpoint-research": "https://research.checkpoint.com/feed/",
        "rapid7": "https://blog.rapid7.com/rss/",
        "dfir-report": "https://thedfirreport.com/feed/",
        "malwarebytes": "https://www.malwarebytes.com/blog/feed/index.xml",
        "microsoft-security": "https://www.microsoft.com/en-us/security/blog/feed/",
        "mandiant": "https://www.mandiant.com/resources/blog/rss.xml",
        "grahamcluley": "https://grahamcluley.com/feed/",
        "darkreading": "https://www.darkreading.com/rss.xml",
        "securityweek": "https://feeds.feedburner.com/securityweek",
        "helpnetsecurity": "https://www.helpnetsecurity.com/feed/",
        "theregister-security": "https://www.theregister.com/security/headlines.atom",
        "therecord": "https://therecord.media/feed/",
        "tenable": "https://www.tenable.com/blog/feed",
        "sucuri": "https://blog.sucuri.net/feed",
        "sentinelone": "https://www.sentinelone.com/blog/feed/",
        "crowdstrike": "https://www.crowdstrike.com/en-us/blog/feed/",
        "ncsc-uk": "https://www.ncsc.gov.uk/api/1/services/v1/all-rss-feed.xml",
        "fortinet": "https://feeds.fortinet.com/fortinet/blog/threat-research",
    }

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
