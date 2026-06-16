from sentinel.api.limits import RateLimiter


def test_rate_limiter_blocks_past_the_window_quota() -> None:
    limiter = RateLimiter(max_requests=2, window_seconds=10)
    # Inject a fixed clock so the test is deterministic (no sleeping).
    assert limiter.allow("a", now=100.0) is True
    assert limiter.allow("a", now=100.5) is True
    assert limiter.allow("a", now=101.0) is False  # third hit inside the window


def test_rate_limiter_window_slides() -> None:
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    assert limiter.allow("a", now=100.0) is True
    assert limiter.allow("a", now=105.0) is False  # still inside the 10s window
    assert limiter.allow("a", now=111.0) is True  # earlier hit has aged out


def test_rate_limiter_isolates_keys() -> None:
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    assert limiter.allow("a", now=100.0) is True
    assert limiter.allow("b", now=100.0) is True  # a different client is unaffected
    assert limiter.allow("a", now=100.0) is False


def test_rate_limiter_sweeps_stale_keys() -> None:
    limiter = RateLimiter(max_requests=1, window_seconds=1)
    for i in range(10_050):  # cross the sweep threshold with stale, aged-out keys
        limiter.allow(f"client-{i}", now=0.0)
    # A much later request triggers the sweep; the table must not grow unbounded.
    limiter.allow("fresh", now=10_000.0)
    assert len(limiter._hits) < 10_050
