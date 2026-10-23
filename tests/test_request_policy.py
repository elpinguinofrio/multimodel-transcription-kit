"""send_with_limits: one retry policy for every provider."""
from unittest.mock import MagicMock, patch

import pytest
import requests

from multimodel_transcription_kit.rate_limiter import (
    Limits, QuotaExhausted, RateLimiter, RateLimitTimeout,
)
from multimodel_transcription_kit.request_policy import (
    is_daily_quota, retry_delay_from, send_with_limits,
)

DAILY_429 = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
        {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "21s"}]}}
MINUTE_429 = {"error": {"code": 429, "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
        {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "43s"}]}}


def resp(status, body=None, headers=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.headers = headers or {}
    if body is None:
        r.json.side_effect = ValueError("no json")
    else:
        r.json.return_value = body
    return r


class Clock:
    def __init__(self):
        self.now = 1_760_000_000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


def test_retry_delay_reads_header_then_body():
    assert retry_delay_from(resp(429, headers={"retry-after": "7"})) == 7.0
    assert retry_delay_from(resp(429, MINUTE_429)) == 43.0
    assert retry_delay_from(resp(429)) is None


def test_daily_quota_detected_only_for_per_day_violations():
    assert is_daily_quota(resp(429, DAILY_429))
    assert not is_daily_quota(resp(429, MINUTE_429))
    assert not is_daily_quota(resp(429))


def test_minute_429_blocks_key_for_server_delay_then_succeeds():
    clock = Clock()
    rl = RateLimiter({}, clock=clock, sleep=clock.sleep)
    send = MagicMock(side_effect=[resp(429, MINUTE_429), resp(200, {})])
    res = send_with_limits(send, limiter=rl, key="m", sleep=clock.sleep)
    assert res.status_code == 200
    assert clock.slept == [pytest.approx(43.0)]


def test_daily_429_raises_quota_exhausted_and_blocks_other_callers():
    clock = Clock()
    rl = RateLimiter({}, clock=clock, sleep=clock.sleep)
    send = MagicMock(return_value=resp(429, DAILY_429))
    with pytest.raises(QuotaExhausted):
        send_with_limits(send, limiter=rl, key="m", sleep=clock.sleep)
    assert send.call_count == 1
    with pytest.raises(QuotaExhausted):
        rl.acquire("m")


def test_daily_429_without_limiter_still_raises_quota_exhausted():
    send = MagicMock(return_value=resp(429, DAILY_429))
    with pytest.raises(QuotaExhausted):
        send_with_limits(send, limiter=None, key="m", sleep=lambda s: None)


def test_limiter_slot_is_taken_before_every_attempt():
    clock = Clock()
    rl = RateLimiter({"m": Limits(rpd=10)}, clock=clock, sleep=clock.sleep)
    send = MagicMock(side_effect=[resp(503), resp(200, {})])
    send_with_limits(send, limiter=rl, key="m", base_delay=0, sleep=clock.sleep)
    assert rl.usage("m")["today"] == 2


def test_endless_429_ends_in_timeout_never_a_quiet_429():
    clock = Clock()
    rl = RateLimiter({}, clock=clock, sleep=clock.sleep, max_wait=10_000_000)
    send = MagicMock(return_value=resp(429, MINUTE_429))
    with pytest.raises(RateLimitTimeout):
        send_with_limits(send, limiter=rl, key="m", sleep=clock.sleep)
    assert send.call_count == 21


def test_exhausted_429_without_limiter_raises_timeout():
    send = MagicMock(return_value=resp(429, MINUTE_429))
    with pytest.raises(RateLimitTimeout):
        send_with_limits(send, limiter=None, key="m", max_retries=2, sleep=lambda s: None)
    assert send.call_count == 3


@pytest.mark.parametrize("body", [
    {"error": {"details": [{"violations": [{"quotaMetric": "x/requests_per_day"}]}]}},
    {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDay"}]}]}},
    {"error": {"code": 429, "message": "Quota exceeded: requests per day for this model"}},
])
def test_daily_quota_detected_in_every_known_shape(body):
    assert is_daily_quota(resp(429, body))


def test_per_minute_message_is_not_daily():
    assert not is_daily_quota(resp(429, {"error": {"message": "requests per minute exceeded"}}))


def test_gemini_api_routes_requests_through_limiter():
    from multimodel_transcription_kit.gemini_client import GeminiAPI
    clock = Clock()
    rl = RateLimiter({"gemini-3.5-flash-lite": Limits(rpd=1)}, clock=clock, sleep=clock.sleep)
    api = GeminiAPI(rate_limiter=rl)
    ok = resp(200, {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
    with patch("requests.post", return_value=ok):
        assert api.generate_text("x", model="gemini-3.5-flash-lite") == "hi"
        with pytest.raises(QuotaExhausted):
            api.generate_text("x", model="gemini-3.5-flash-lite")


def test_flash_lite_is_a_known_model():
    from multimodel_transcription_kit.gemini_client import GeminiModel
    assert GeminiModel("gemini-3.5-flash-lite").value == "gemini-3.5-flash-lite"
