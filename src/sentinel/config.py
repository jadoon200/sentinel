from functools import lru_cache

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

    http_timeout_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
