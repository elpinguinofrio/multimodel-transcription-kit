"""One retry policy for every provider: wait for a slot, back off on 429/5xx."""

import logging
import random
import re
import time
import uuid
from typing import Callable, Optional

import requests

from .rate_limiter import QuotaExhausted, RateLimiter, RateLimitTimeout, next_quota_reset
from .usage import track

logger = logging.getLogger("request_policy")

RETRYABLE_5XX = {500, 502, 503, 504}
# Upper bound on 429 rounds; each round already waits the server's delay.
MAX_RATE_LIMIT_ROUNDS = 20


def _error(response) -> dict:
    try:
        body = response.json()
    except Exception:
        return {}
    if not isinstance(body, dict) or not isinstance(body.get("error"), dict):
        return {}
    return body["error"]


def _error_details(response) -> list:
    details = _error(response).get("details")
    return [d for d in details if isinstance(d, dict)] if isinstance(details, list) else []


_DAILY = re.compile(r"per[ _-]?day|daily", re.IGNORECASE)


def retry_delay_from(response) -> Optional[float]:
    """Seconds the server asked us to wait: Retry-After header, else Gemini RetryInfo."""
    headers = getattr(response, "headers", None) or {}
    for name in ("Retry-After", "retry-after"):
        value = headers.get(name)
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    for detail in _error_details(response):
        match = re.fullmatch(r"([\d.]+)s", str(detail.get("retryDelay", "")))
        if match:
            return float(match.group(1))
    return None


def is_daily_quota(response) -> bool:
    """Gemini names the violated quota; a per-day quota will not free up today."""
    for detail in _error_details(response):
        for violation in detail.get("violations") or []:
            if not isinstance(violation, dict):
                continue
            if _DAILY.search(f"{violation.get('quotaId', '')} {violation.get('quotaMetric', '')}"):
                return True
    return bool(_DAILY.search(str(_error(response).get("message", ""))))


def backoff(attempt: int, base_delay: float, max_delay: float) -> float:
    return min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 0.5)


def send_with_limits(send: Callable[[], requests.Response], *, limiter: Optional[RateLimiter],
                     key: str, tokens: int = 0, max_retries: int = 4, base_delay: float = 2.0,
                     max_delay: float = 60.0, sleep=None, usage_log=None,
                     provider="unknown", operation="transcription") -> requests.Response:
    """Send once per free slot; retry 429, 5xx and connection errors.

    Returns the final response (the caller raises on non-200). Raises
    QuotaExhausted when the daily quota is gone, RateLimitTimeout when a slot
    is further away than the limiter's max_wait or 429s never stop. A 429 is
    never returned, so a caller cannot mistake it for an ordinary failure.
    """
    sleep = sleep or time.sleep
    attempt = 0
    rate_limited = 0
    request_id = uuid.uuid4().hex
    sent_attempts = 0
    while True:
        if limiter is not None:
            limiter.acquire(key, tokens)
        try:
            sent_attempts += 1
            with track(usage_log, provider, key, operation,
                       request_id=request_id, attempt=sent_attempts) as event:
                response = send()
                event["http_status"] = response.status_code
                if usage_log is not None:
                    try:
                        event["response"] = response.json()
                    except (ValueError, TypeError):
                        pass
        except requests.exceptions.RequestException as exc:
            if attempt >= max_retries:
                raise
            wait = backoff(attempt, base_delay, max_delay)
            logger.warning("%s: request error (%s), retrying in %.1fs", key, exc, wait)
            sleep(wait)
            attempt += 1
            continue

        if response.status_code == 429:
            if is_daily_quota(response):
                reason = "server reported the daily quota used up"
                if limiter is not None:
                    raise limiter.block_until_reset(key, reason)
                raise QuotaExhausted(key, next_quota_reset(time.time()), reason)
            delay = retry_delay_from(response)
            if limiter is not None:
                if rate_limited >= MAX_RATE_LIMIT_ROUNDS:
                    raise RateLimitTimeout(f"{key}: still rate limited after {rate_limited} waits")
                rate_limited += 1
                wait = delay if delay is not None else backoff(rate_limited, base_delay, max_delay)
                logger.warning("%s: 429, all callers wait %.1fs", key, wait)
                limiter.block(key, wait)
                continue
            if attempt >= max_retries:
                raise RateLimitTimeout(f"{key}: still rate limited after {attempt + 1} attempts")
            wait = min(delay, max_delay) if delay is not None else backoff(attempt, base_delay, max_delay)
            logger.warning("%s: 429, retrying in %.1fs", key, wait)
            sleep(wait)
            attempt += 1
            continue

        if response.status_code in RETRYABLE_5XX and attempt < max_retries:
            wait = backoff(attempt, base_delay, max_delay)
            logger.warning("%s: HTTP %d, retrying in %.1fs", key, response.status_code, wait)
            sleep(wait)
            attempt += 1
            continue
        return response
