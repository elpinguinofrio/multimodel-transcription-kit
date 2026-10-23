"""
Tests for GeminiAPI retry/backoff behaviour.

TDD Red Phase: these fail until `_post_with_retry` + retry config land in
gemini_client.py. They cover the gap codex flagged: a transient 429 / 5xx /
connection reset on a transcription call was raised immediately with no retry,
which on a chunked recording burns a whole segment into a permanent hole.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests


def _load_gemini():
    from importlib import import_module
    return import_module("multimodel_transcription_kit.gemini_client")


_gemini = _load_gemini()
GeminiAPI = _gemini.GeminiAPI


def _api(**kwargs):
    # GEMINI_BASE_URL set => offline construction (no real key, no network at init).
    with patch.dict("os.environ", {"GEMINI_BASE_URL": "http://127.0.0.1:8300"}):
        return GeminiAPI(**kwargs)


def _resp(status, headers=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.headers = headers or {}
    return r


class TestPostWithRetry:
    def test_retries_on_429_then_succeeds(self):
        api = _api(max_retries=3, retry_base_delay=0.0)
        post = MagicMock(side_effect=[_resp(429), _resp(200)])
        with patch("requests.post", post), patch("time.sleep") as sleep:
            res = api._post_with_retry("http://x", {}, timeout=1, headers=None)
        assert res.status_code == 200
        assert post.call_count == 2
        assert sleep.called

    def test_retries_on_transient_5xx_then_succeeds(self):
        api = _api(max_retries=3, retry_base_delay=0.0)
        post = MagicMock(side_effect=[_resp(503), _resp(200)])
        with patch("requests.post", post), patch("time.sleep"):
            res = api._post_with_retry("http://x", {}, timeout=1, headers=None)
        assert res.status_code == 200
        assert post.call_count == 2

    def test_retries_on_connection_error_then_succeeds(self):
        api = _api(max_retries=3, retry_base_delay=0.0)
        post = MagicMock(side_effect=[requests.exceptions.ConnectionError("reset"), _resp(200)])
        with patch("requests.post", post), patch("time.sleep"):
            res = api._post_with_retry("http://x", {}, timeout=1, headers=None)
        assert res.status_code == 200
        assert post.call_count == 2

    def test_no_retry_on_permanent_400(self):
        api = _api(max_retries=3, retry_base_delay=0.0)
        post = MagicMock(return_value=_resp(400))
        with patch("requests.post", post), patch("time.sleep") as sleep:
            res = api._post_with_retry("http://x", {}, timeout=1, headers=None)
        assert res.status_code == 400
        assert post.call_count == 1
        assert not sleep.called

    def test_honors_retry_after_header(self):
        api = _api(max_retries=2, retry_base_delay=0.0, max_retry_delay=60.0)
        post = MagicMock(side_effect=[_resp(429, {"Retry-After": "7"}), _resp(200)])
        with patch("requests.post", post), patch("time.sleep") as sleep:
            api._post_with_retry("http://x", {}, timeout=1, headers=None)
        # First (and only) backoff should wait ~7s, honoring the header.
        assert sleep.call_args_list[0].args[0] >= 7.0

    def test_exhausted_429_raises_rate_limit_timeout(self):
        # 0.2.0: an exhausted 429 raises instead of returning, so a chunked
        # transcript can never merge around a rate-limited hole.
        from multimodel_transcription_kit.rate_limiter import RateLimitTimeout
        api = _api(max_retries=2, retry_base_delay=0.0)
        post = MagicMock(return_value=_resp(429))
        with patch("requests.post", post), patch("time.sleep"):
            with pytest.raises(RateLimitTimeout):
                api._post_with_retry("http://x", {}, timeout=1, headers=None)
        # 1 initial attempt + 2 retries
        assert post.call_count == 3

    def test_reraises_connection_error_after_exhaustion(self):
        api = _api(max_retries=1, retry_base_delay=0.0)
        post = MagicMock(side_effect=requests.exceptions.ConnectionError("reset"))
        with patch("requests.post", post), patch("time.sleep"):
            try:
                api._post_with_retry("http://x", {}, timeout=1, headers=None)
                raised = False
            except requests.exceptions.ConnectionError:
                raised = True
        assert raised
        assert post.call_count == 2  # 1 initial + 1 retry
