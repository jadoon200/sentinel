import os

# torch and lightgbm both vendor libomp on macOS. Loading torch's copy first
# makes lightgbm's first Dataset construction segfault, so force lightgbm's
# runtime to load before any test module imports torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import lightgbm  # noqa: F401
import pytest

from sentinel.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
