"""Tests for web/ratelimit.py — token bucket rate limiting."""

import time

from palmtop.web.ratelimit import RateLimiter


class TestChatRateLimit:
    def test_allows_within_limit(self):
        limiter = RateLimiter(chat_rpm=5, chat_rpd=100)
        # First 5 should be allowed
        for _ in range(5):
            assert limiter.check_chat("user1")

    def test_blocks_over_per_minute_limit(self):
        limiter = RateLimiter(chat_rpm=3, chat_rpd=100)
        # Exhaust the bucket
        for _ in range(3):
            assert limiter.check_chat("user1")
        # 4th should be blocked
        assert not limiter.check_chat("user1")

    def test_blocks_over_daily_limit(self):
        limiter = RateLimiter(chat_rpm=1000, chat_rpd=2)
        assert limiter.check_chat("user1")
        assert limiter.check_chat("user1")
        assert not limiter.check_chat("user1")

    def test_separate_users_have_separate_limits(self):
        limiter = RateLimiter(chat_rpm=1, chat_rpd=100)
        assert limiter.check_chat("user1")
        assert not limiter.check_chat("user1")
        # Different user should still be allowed
        assert limiter.check_chat("user2")


class TestFormRateLimit:
    def test_allows_within_limit(self):
        limiter = RateLimiter(form_rpm=3, form_rpd=10)
        for _ in range(3):
            assert limiter.check_form("user1")

    def test_blocks_over_limit(self):
        limiter = RateLimiter(form_rpm=1, form_rpd=10)
        assert limiter.check_form("user1")
        assert not limiter.check_form("user1")


class TestConcurrency:
    def test_acquire_and_release(self):
        limiter = RateLimiter(max_concurrent=2)
        assert limiter.acquire_stream()
        assert limiter.acquire_stream()
        assert not limiter.acquire_stream()  # at capacity
        limiter.release_stream()
        assert limiter.acquire_stream()  # one slot freed

    def test_release_never_goes_negative(self):
        limiter = RateLimiter(max_concurrent=2)
        limiter.release_stream()
        limiter.release_stream()
        # Should still allow 2 (not 4)
        assert limiter.acquire_stream()
        assert limiter.acquire_stream()
        assert not limiter.acquire_stream()


class TestCleanup:
    def test_removes_stale_buckets(self, monkeypatch):
        limiter = RateLimiter(chat_rpm=10, chat_rpd=100)
        limiter.check_chat("user1")

        # Artificially age the bucket
        for store in (limiter._minute_buckets, limiter._day_buckets):
            for bucket in store.values():
                bucket.last_refill = time.time() - 86401  # >24h ago

        removed = limiter.cleanup()
        assert removed >= 1
        assert len(limiter._minute_buckets) == 0 or len(limiter._day_buckets) == 0
