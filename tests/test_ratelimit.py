from app.ratelimit import RateLimiter


def test_blocks_after_limit_and_recovers_after_window():
    limiter = RateLimiter(max_requests=2, window_seconds=10)
    assert limiter.allow("a", now=0)
    assert limiter.allow("a", now=1)
    assert not limiter.allow("a", now=2)
    assert limiter.allow("a", now=11)  # first hit has expired


def test_each_caller_has_its_own_budget():
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    assert limiter.allow("a", now=0)
    assert limiter.allow("b", now=0)
    assert not limiter.allow("a", now=1)
