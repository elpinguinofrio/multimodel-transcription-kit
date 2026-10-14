# Rate Limits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every provider request made through the core waits for a free slot under the owner's real limits (per minute, tokens per minute, per day), shared across threads and processes, instead of failing with 429.

**Architecture:** A new `RateLimiter` keeps request history in a SQLite file the caller names (in memory if none). One shared send loop, `send_with_limits`, asks the limiter for a slot before every attempt, turns server 429s into a shared back-off, and turns "daily quota used up" into a clear `QuotaExhausted` error. `GeminiAPI`, `GroqAPI`, `transcribe`, `transcribe_large` and the CLI accept the limiter as a new optional argument; nothing existing changes signature.

**Tech Stack:** Python ≥3.10 stdlib (`sqlite3`, `zoneinfo`, `configparser`, `threading`), `requests`, pytest.

**Spec:** `docs/superpowers/specs/rate-limits.md` (limits: `docs/superpowers/specs/gemini-pro-tier-rate-limits.md`)

## Global Constraints

- Importing the package must not call providers, discover configuration or write files (AGENTS.md). The limiter writes only to a `state_path` the caller passes.
- Configuration is supplied explicitly by the caller; the library never searches for `transcription.ini`.
- Public signatures stay compatible: only new optional keyword arguments. Version bumps to 0.2.0 with migration notes for Exocortex and Clip Smith.
- No live provider calls in tests (conftest blocks non-localhost sockets). No benchmark runs: the benchmark is owned by another agent.
- State file must be on a local disk, never `/Volumes/VM-shared` (SMB; SQLite locking is unreliable there).
- Gemini daily quotas reset at midnight America/Los_Angeles.
- Do not touch `benchmark/` files.

## Review Focus

1. A single request bigger than the TPM limit (a 10-min clip is ~19K tokens vs 10K TPM on 3.5 Transcribe) must still go through when the minute window is empty, not wait forever → Task 1 test `test_oversize_request_runs_when_window_is_empty`.
2. Two apps or threads at once must never exceed the limit together → Task 1 tests `test_threads_never_exceed_rpm` and `test_processes_share_one_state_file`.
3. Daily quota used up must stop the whole recording loudly, not leave silent holes in a chunked transcript → Task 3 test `test_quota_exhausted_aborts_chunked_transcription`.
4. Groq retries must resend the audio, not an already-read empty buffer → Task 4 test `test_groq_retry_resends_full_audio`.
5. Limits copied from AI Studio as `250K`, `14.4K` or `Unlimited` must parse, not crash → Task 1 test `test_parse_count_accepts_ai_studio_values`.

## Revisions after Codex plan review (2026-10-14)

These override the task text below where they conflict.

- R1 Exhausted 429s never return a 429: `send_with_limits` raises `RateLimitTimeout` after `MAX_RATE_LIMIT_ROUNDS` (with a limiter) or `max_retries` (without), so chunked transcription aborts instead of merging holes.
- R2 No TPM bypass: a request whose estimate exceeds the model's TPM raises `RequestTooLarge(RateLimitError)` before sending. Replaces `test_oversize_request_runs_when_window_is_empty`.
- R3 SQLite coordinates one machine only. VM and host each get a static share of RPM, TPM and RPD in their own `transcription.ini` (documented in `docs/integration.md`).
- R4 Groq is reactive only (honours `retry-after`, optional rpm/rpd); its limits are unknown and its day boundary is not Pacific. Documented, no proactive token accounting.
- R5 Daily detection checks `quotaId` and `quotaMetric` and `error.message` for per-day wording; a 429 that is not recognised as daily retries and ends in `RateLimitTimeout`, never silently.
- R6 SQLite lock contention: short connect timeout, `database is locked` retried until the caller's deadline, then `RateLimitTimeout`. `usage()` uses a read transaction.
- R7 Process test: 6 processes released together by a barrier, rpm=3, `max_wait=0` → exactly 3 succeed.
- R8 Quickstart example (`[provider.*]` schema) is not wired; only callers that pass a limiter are protected. Stated in the guide.
- R9 Spec "Done when": owner instructed no benchmark runs (another agent owns the benchmark), so the 120 s run is replaced by a one-request smoke on a generated clip.
- R10 Merge failure no longer drops words: when the boundary merge returns empty, `merge_adjacent_chunks` keeps `tail_a + head_b` (possible duplicate overlap, no loss).
- R11 `api if api is not None else GeminiAPI()`.

After the Codex code review:

- C1 `transcribe_large(..., model=)` uses one model for short files, chunks and merges, so a configured limit covers every request (default keeps the old models).
- C2 SQLite lock waits are bounded by the caller's deadline (0.5 s floor); `block()` lock failures become `RateLimitTimeout`.
- C3 A configured limit of 0 raises `QuotaExhausted` instead of `IndexError`.
- C4 Quickstart example builds a limiter from its `[provider.*]` sections (`from_parser(section_prefix=...)`) and passes it to every client.
- C5 A quota error during merging still cleans up chunk files.
- C6 The merge fallback test asserts the exact result `a + " " + b`.

---

### Task 1: `RateLimiter` with shared SQLite state

**Files:**
- Create: `src/multimodal_transcription_core/rate_limiter.py`
- Test: `tests/test_rate_limiter.py`

**Interfaces:**
- Produces:
  - `Limits(rpm: int|None, tpm: int|None, rpd: int|None)` frozen dataclass
  - `RateLimitError(Exception)`, `QuotaExhausted(RateLimitError)` with `.key: str`, `.resets_at: float`; `RateLimitTimeout(RateLimitError)`
  - `RateLimiter(limits: dict[str, Limits], state_path=None, max_wait=3600.0, clock=time.time, sleep=time.sleep)`
    - `.acquire(key: str, tokens: int = 0) -> None` (blocks; raises `QuotaExhausted` / `RateLimitTimeout`)
    - `.block(key: str, seconds: float, reason: str = ...) -> None`
    - `.block_until_reset(key: str, reason: str) -> QuotaExhausted` (returns the error for the caller to raise)
    - `.usage(key: str) -> dict` with `today`, `last_minute`, `tokens_last_minute`
    - `RateLimiter.from_config(path, **kw)` and `RateLimiter.from_parser(parser, **kw)`
  - `estimate_audio_tokens(duration_sec: float) -> int`, `estimate_text_tokens(text: str) -> int`
  - `next_quota_reset(now: float) -> float`, `parse_count(value: str|None) -> int|None`

- [ ] **Step 1: Write the failing tests** — `tests/test_rate_limiter.py`

```python
"""RateLimiter: waits for a free slot, shares counts, stops on daily quota."""
import configparser
import multiprocessing
import threading
from datetime import datetime

import pytest

from multimodal_transcription_core.rate_limiter import (
    Limits, QuotaExhausted, RateLimiter, RateLimitTimeout, QUOTA_DAY_TZ,
    estimate_audio_tokens, next_quota_reset, parse_count,
)


class FakeClock:
    def __init__(self, now=1_760_000_000.0):
        self.now = now
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def limiter(limits, clock, **kw):
    return RateLimiter(limits, clock=clock, sleep=clock.sleep, **kw)


def test_requests_under_rpm_do_not_wait():
    clock = FakeClock()
    rl = limiter({"m": Limits(rpm=3)}, clock)
    for _ in range(3):
        rl.acquire("m")
    assert clock.slept == []


def test_request_over_rpm_waits_for_oldest_to_leave_window():
    clock = FakeClock()
    rl = limiter({"m": Limits(rpm=2)}, clock)
    rl.acquire("m")
    clock.now += 10
    rl.acquire("m")
    rl.acquire("m")
    assert clock.slept == [pytest.approx(50.0)]


def test_tpm_waits_until_enough_tokens_free():
    clock = FakeClock()
    rl = limiter({"m": Limits(tpm=1000)}, clock)
    rl.acquire("m", tokens=600)
    clock.now += 5
    rl.acquire("m", tokens=600)
    assert clock.slept == [pytest.approx(55.0)]


def test_oversize_request_runs_when_window_is_empty():
    clock = FakeClock()
    rl = limiter({"m": Limits(tpm=10_000)}, clock)
    rl.acquire("m", tokens=19_700)
    assert clock.slept == []


def test_daily_limit_raises_with_reset_time():
    clock = FakeClock()
    rl = limiter({"m": Limits(rpd=2)}, clock)
    rl.acquire("m")
    rl.acquire("m")
    with pytest.raises(QuotaExhausted) as err:
        rl.acquire("m")
    assert err.value.resets_at == next_quota_reset(clock.now)
    assert clock.slept == []


def test_daily_count_resets_at_pacific_midnight():
    clock = FakeClock()
    rl = limiter({"m": Limits(rpd=1)}, clock)
    rl.acquire("m")
    clock.now = next_quota_reset(clock.now) + 1
    rl.acquire("m")


def test_next_quota_reset_is_pacific_midnight():
    reset = datetime.fromtimestamp(next_quota_reset(1_760_000_000.0), QUOTA_DAY_TZ)
    assert (reset.hour, reset.minute, reset.second) == (0, 0, 0)


def test_unknown_key_is_not_limited():
    clock = FakeClock()
    rl = limiter({}, clock)
    for _ in range(100):
        rl.acquire("other")
    assert clock.slept == []


def test_wait_longer_than_max_wait_raises_timeout():
    clock = FakeClock()
    rl = limiter({"m": Limits(rpm=1)}, clock, max_wait=10)
    rl.acquire("m")
    with pytest.raises(RateLimitTimeout):
        rl.acquire("m")


def test_server_block_makes_next_acquire_wait():
    clock = FakeClock()
    rl = limiter({}, clock)
    rl.block("m", 30)
    rl.acquire("m")
    assert clock.slept == [pytest.approx(30.0)]


def test_block_until_reset_stops_until_midnight():
    clock = FakeClock()
    rl = limiter({}, clock)
    err = rl.block_until_reset("m", "server said PerDay")
    assert isinstance(err, QuotaExhausted)
    with pytest.raises(QuotaExhausted):
        rl.acquire("m")
    clock.now = next_quota_reset(clock.now) + 1
    rl.acquire("m")


def test_usage_reports_counts():
    clock = FakeClock()
    rl = limiter({"m": Limits(rpd=500)}, clock)
    rl.acquire("m", tokens=100)
    rl.acquire("m", tokens=50)
    assert rl.usage("m") == {"today": 2, "last_minute": 2, "tokens_last_minute": 150}


def test_threads_never_exceed_rpm():
    rl = RateLimiter({"m": Limits(rpm=5)}, max_wait=0)
    outcomes = []

    def worker():
        try:
            rl.acquire("m")
            outcomes.append("ok")
        except RateLimitTimeout:
            outcomes.append("timeout")

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("ok") == 5


def _acquire_twice(path):
    rl = RateLimiter({"m": Limits(rpm=4)}, state_path=path, max_wait=0)
    rl.acquire("m")
    rl.acquire("m")


def test_processes_share_one_state_file(tmp_path):
    path = tmp_path / "state" / "limits.sqlite"
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_acquire_twice, args=(str(path),)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    assert [p.exitcode for p in procs] == [0, 0]
    rl = RateLimiter({"m": Limits(rpm=4)}, state_path=path, max_wait=0)
    with pytest.raises(RateLimitTimeout):
        rl.acquire("m")


def test_parse_count_accepts_ai_studio_values():
    assert parse_count("15") == 15
    assert parse_count("250K") == 250_000
    assert parse_count("14.4K") == 14_400
    assert parse_count("1M") == 1_000_000
    assert parse_count("Unlimited") is None
    assert parse_count("") is None
    assert parse_count(None) is None


def test_from_parser_reads_model_sections(tmp_path):
    parser = configparser.ConfigParser()
    parser.read_string(f"""
[limits]
state_file = {tmp_path}/s.sqlite
max_wait_seconds = 120

[model.flash-lite]
kind = gemini
model = gemini-3.5-flash-lite
rpm = 15
tpm = 250K
rpd = 500

[model.local]
kind = http
""")
    rl = RateLimiter.from_parser(parser)
    assert rl.limits == {"gemini-3.5-flash-lite": Limits(rpm=15, tpm=250_000, rpd=500)}
    assert rl.max_wait == 120
    assert (tmp_path / "s.sqlite").exists()


def test_estimate_audio_tokens_uses_32_per_second():
    assert estimate_audio_tokens(600) == 600 * 32 + 500
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_rate_limiter.py -q`
Expected: collection error `ModuleNotFoundError: ... rate_limiter`

- [ ] **Step 3: Implement** — `src/multimodal_transcription_core/rate_limiter.py`

```python
"""Shared request queue that keeps every caller under provider limits.

Processes that pass the same state file share one set of counters, so two
apps on one API key wait for each other instead of tripping a 429. Keep the
state file on a local disk: SQLite locking is unreliable on SMB/NFS mounts.
"""

import configparser
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("rate_limiter")

# Gemini daily quotas reset at midnight Pacific time.
QUOTA_DAY_TZ = ZoneInfo("America/Los_Angeles")
# Gemini counts audio as 32 tokens per second; the allowance covers the prompt.
AUDIO_TOKENS_PER_SECOND = 32
PROMPT_TOKEN_ALLOWANCE = 500
WINDOW_SECONDS = 60.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (key TEXT NOT NULL, ts REAL NOT NULL, tokens INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS requests_key_ts ON requests (key, ts);
CREATE TABLE IF NOT EXISTS blocks (key TEXT PRIMARY KEY, until REAL NOT NULL,
                                   daily INTEGER NOT NULL, reason TEXT NOT NULL);
"""


@dataclass(frozen=True)
class Limits:
    rpm: Optional[int] = None
    tpm: Optional[int] = None
    rpd: Optional[int] = None


class RateLimitError(Exception):
    """A request could not be sent within the configured limits."""


class QuotaExhausted(RateLimitError):
    """The daily quota for this model is used up; retrying today is pointless."""

    def __init__(self, key: str, resets_at: float, reason: str):
        self.key = key
        self.resets_at = resets_at
        local = datetime.fromtimestamp(resets_at, QUOTA_DAY_TZ)
        super().__init__(f"{key}: daily quota used up ({reason}); resets {local:%Y-%m-%d %H:%M %Z}")


class RateLimitTimeout(RateLimitError):
    """Waiting for a free slot would take longer than max_wait."""


def quota_day_start(now: float) -> float:
    local = datetime.fromtimestamp(now, QUOTA_DAY_TZ)
    return local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def next_quota_reset(now: float) -> float:
    local = datetime.fromtimestamp(now, QUOTA_DAY_TZ)
    tomorrow = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return tomorrow.timestamp()


def estimate_audio_tokens(duration_sec: float) -> int:
    return int(max(duration_sec or 0, 0) * AUDIO_TOKENS_PER_SECOND) + PROMPT_TOKEN_ALLOWANCE


def estimate_text_tokens(text: str) -> int:
    return len(text or "") // 3 + PROMPT_TOKEN_ALLOWANCE


def parse_count(value: Optional[str]) -> Optional[int]:
    """'15', '250K', '14.4K', '1M' -> int; '', 'Unlimited', None -> no limit."""
    text = (value or "").strip().upper()
    if not text or text == "UNLIMITED":
        return None
    scale = {"K": 1_000, "M": 1_000_000}.get(text[-1], 1)
    number = text[:-1] if scale > 1 else text
    return int(round(float(number) * scale))


class RateLimiter:
    def __init__(self, limits: Dict[str, Limits], state_path=None, max_wait: float = 3600.0,
                 clock=time.time, sleep=time.sleep):
        self.limits = dict(limits)
        self.max_wait = max_wait
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        if state_path is None:
            self._path = None
            self._shared = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
            self._shared.executescript(_SCHEMA)
        else:
            self._path = Path(state_path).expanduser()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._shared = None
            db = sqlite3.connect(self._path, timeout=60, isolation_level=None)
            try:
                db.executescript(_SCHEMA)
            finally:
                db.close()

    # ───────────────────────── config ──────────────────────────

    @classmethod
    def from_config(cls, path, **kwargs) -> "RateLimiter":
        """Build from an INI file the caller names explicitly."""
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"No limits config at {config_path}")
        parser = configparser.ConfigParser()
        parser.read(config_path, encoding="utf-8")
        return cls.from_parser(parser, **kwargs)

    @classmethod
    def from_parser(cls, parser: configparser.ConfigParser, **kwargs) -> "RateLimiter":
        """[limits] state_file/max_wait_seconds and rpm/tpm/rpd in each [model.<name>]."""
        limits = {}
        for section in parser.sections():
            if not section.startswith("model."):
                continue
            options = parser[section]
            found = Limits(parse_count(options.get("rpm")), parse_count(options.get("tpm")),
                           parse_count(options.get("rpd")))
            if found != Limits():
                limits[options.get("model", section[len("model."):])] = found
        kwargs.setdefault("state_path", parser.get("limits", "state_file", fallback=None))
        kwargs.setdefault("max_wait", parser.getfloat("limits", "max_wait_seconds", fallback=3600.0))
        return cls(limits, **kwargs)

    # ───────────────────────── storage ──────────────────────────

    @contextmanager
    def _transaction(self):
        with self._lock:
            db = self._shared or sqlite3.connect(self._path, timeout=60, isolation_level=None)
            try:
                db.execute("BEGIN IMMEDIATE")
                try:
                    yield db
                except BaseException:
                    db.execute("ROLLBACK")
                    raise
                db.execute("COMMIT")
            finally:
                if db is not self._shared:
                    db.close()

    # ───────────────────────── public api ──────────────────────────

    def acquire(self, key: str, tokens: int = 0) -> None:
        """Block until a request for `key` fits every limit, then record it."""
        deadline = self._clock() + self.max_wait
        while True:
            with self._transaction() as db:
                wait = self._reserve(db, key, tokens)
            if wait <= 0:
                return
            if self._clock() + wait > deadline:
                raise RateLimitTimeout(
                    f"{key}: next free slot is {wait:.0f}s away, over max_wait={self.max_wait:.0f}s")
            logger.info("%s: waiting %.1fs for a free slot", key, wait)
            self._sleep(wait)

    def block(self, key: str, seconds: float, reason: str = "server asked to slow down") -> None:
        """Every process sharing this state holds `key` for `seconds`."""
        self._set_block(key, self._clock() + max(seconds, 0.0), False, reason)

    def block_until_reset(self, key: str, reason: str) -> QuotaExhausted:
        """Server says the day is used up: hold `key` until the quota resets."""
        until = next_quota_reset(self._clock())
        self._set_block(key, until, True, reason)
        return QuotaExhausted(key, until, reason)

    def usage(self, key: str) -> Dict[str, int]:
        now = self._clock()
        with self._transaction() as db:
            (today,) = db.execute("SELECT COUNT(*) FROM requests WHERE key=? AND ts>=?",
                                  (key, quota_day_start(now))).fetchone()
            minute, tokens = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM requests WHERE key=? AND ts>?",
                (key, now - WINDOW_SECONDS)).fetchone()
        return {"today": today, "last_minute": minute, "tokens_last_minute": tokens}

    # ───────────────────────── internals ──────────────────────────

    def _set_block(self, key, until, daily, reason):
        with self._transaction() as db:
            row = db.execute("SELECT until FROM blocks WHERE key=?", (key,)).fetchone()
            if row and row[0] >= until:
                return
            db.execute("INSERT OR REPLACE INTO blocks (key, until, daily, reason) VALUES (?, ?, ?, ?)",
                       (key, until, int(daily), reason))

    def _reserve(self, db, key, tokens) -> float:
        """Record the request and return 0, or return seconds to wait."""
        now = self._clock()
        row = db.execute("SELECT until, daily, reason FROM blocks WHERE key=?", (key,)).fetchone()
        if row and row[0] > now:
            if row[1]:
                raise QuotaExhausted(key, row[0], row[2])
            return row[0] - now
        day_start = quota_day_start(now)
        db.execute("DELETE FROM requests WHERE ts < ?", (min(day_start, now - WINDOW_SECONDS),))
        limits = self.limits.get(key)
        if limits is not None:
            if limits.rpd is not None:
                (today,) = db.execute("SELECT COUNT(*) FROM requests WHERE key=? AND ts>=?",
                                      (key, day_start)).fetchone()
                if today >= limits.rpd:
                    raise QuotaExhausted(key, next_quota_reset(now), f"{today}/{limits.rpd} requests today")
            window = db.execute("SELECT ts, tokens FROM requests WHERE key=? AND ts>? ORDER BY ts",
                                (key, now - WINDOW_SECONDS)).fetchall()
            if limits.rpm is not None and len(window) >= limits.rpm:
                return window[len(window) - limits.rpm][0] + WINDOW_SECONDS - now
            if limits.tpm is not None and window:
                remaining = sum(t for _, t in window)
                if remaining + tokens > limits.tpm:
                    for ts, used in window:
                        remaining -= used
                        if remaining + tokens <= limits.tpm or remaining == 0:
                            return ts + WINDOW_SECONDS - now
        db.execute("INSERT INTO requests (key, ts, tokens) VALUES (?, ?, ?)", (key, now, tokens))
        return 0.0
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_rate_limiter.py -q`
Expected: all pass.

- [ ] **Step 5: Commit** (only when the owner authorizes commits)

```bash
git add src/multimodal_transcription_core/rate_limiter.py tests/test_rate_limiter.py
git commit -m "feat: shared rate limiter with per-minute, token and daily limits"
```

---

### Task 2: Shared send loop `send_with_limits`, used by `GeminiAPI`

**Files:**
- Create: `src/multimodal_transcription_core/request_policy.py`
- Modify: `src/multimodal_transcription_core/gemini_client.py` (`GeminiModel`, `__init__`, `_post_with_retry`, `speech_to_text` throttle block ~line 679-704, `generate_text` ~line 830-870)
- Test: `tests/test_request_policy.py`, existing `tests/test_gemini_client.py` must stay green

**Interfaces:**
- Consumes: `RateLimiter.acquire/block/block_until_reset`, `QuotaExhausted`, `next_quota_reset`, `estimate_audio_tokens`, `estimate_text_tokens` (Task 1)
- Produces:
  - `send_with_limits(send: Callable[[], Response], *, limiter: RateLimiter|None, key: str, tokens: int = 0, max_retries: int = 4, base_delay: float = 2.0, max_delay: float = 60.0, sleep=None) -> Response`
  - `retry_delay_from(response) -> float|None`, `is_daily_quota(response) -> bool`
  - `GeminiAPI(..., rate_limiter: RateLimiter|None = None)`; `GeminiModel.GEMINI_3_5_FLASH_LITE = "gemini-3.5-flash-lite"`
  - `GeminiAPI._post_with_retry(url, payload, timeout, headers=None, limit_key=None, tokens=0, max_retries=None)`

- [ ] **Step 1: Write the failing tests** — `tests/test_request_policy.py`

```python
"""send_with_limits: one retry policy for every provider."""
from unittest.mock import MagicMock, patch

import pytest
import requests

from multimodal_transcription_core.rate_limiter import Limits, QuotaExhausted, RateLimiter
from multimodal_transcription_core.request_policy import (
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


def test_endless_429_gives_up_and_returns_response():
    clock = Clock()
    rl = RateLimiter({}, clock=clock, sleep=clock.sleep, max_wait=10_000_000)
    send = MagicMock(return_value=resp(429, MINUTE_429))
    res = send_with_limits(send, limiter=rl, key="m", sleep=clock.sleep)
    assert res.status_code == 429
    assert send.call_count == 21


def test_gemini_api_routes_requests_through_limiter():
    from multimodal_transcription_core.gemini_client import GeminiAPI
    clock = Clock()
    rl = RateLimiter({"gemini-3.5-flash-lite": Limits(rpd=1)}, clock=clock, sleep=clock.sleep)
    api = GeminiAPI(rate_limiter=rl)
    ok = resp(200, {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
    with patch("requests.post", return_value=ok):
        assert api.generate_text("x", model="gemini-3.5-flash-lite") == "hi"
        with pytest.raises(QuotaExhausted):
            api.generate_text("x", model="gemini-3.5-flash-lite")


def test_flash_lite_is_a_known_model():
    from multimodal_transcription_core.gemini_client import GeminiModel
    assert GeminiModel("gemini-3.5-flash-lite").value == "gemini-3.5-flash-lite"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_request_policy.py -q`
Expected: `ModuleNotFoundError: ... request_policy`

- [ ] **Step 3: Implement `request_policy.py`**

```python
"""One retry policy for every provider: wait for a slot, back off on 429/5xx."""

import logging
import random
import re
import time
from typing import Callable, Optional

import requests

from .rate_limiter import QuotaExhausted, RateLimiter, next_quota_reset

logger = logging.getLogger("request_policy")

RETRYABLE_5XX = {500, 502, 503, 504}
# Upper bound on 429 rounds; each round already waits the server's delay.
MAX_RATE_LIMIT_ROUNDS = 20


def _error_details(response) -> list:
    try:
        body = response.json()
    except Exception:
        return []
    if not isinstance(body, dict) or not isinstance(body.get("error"), dict):
        return []
    details = body["error"].get("details")
    return details if isinstance(details, list) else []


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
    """Gemini names the violated quota; a PerDay quota will not free up today."""
    for detail in _error_details(response):
        for violation in detail.get("violations") or []:
            if "perday" in str(violation.get("quotaId", "")).lower():
                return True
    return False


def backoff(attempt: int, base_delay: float, max_delay: float) -> float:
    return min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 0.5)


def send_with_limits(send: Callable[[], requests.Response], *, limiter: Optional[RateLimiter],
                     key: str, tokens: int = 0, max_retries: int = 4, base_delay: float = 2.0,
                     max_delay: float = 60.0, sleep=None) -> requests.Response:
    """Send once per free slot; retry 429, 5xx and connection errors.

    Returns the final response (the caller raises on non-200). Raises
    QuotaExhausted when the daily quota is gone, RateLimitTimeout when a slot
    is further away than the limiter's max_wait.
    """
    sleep = sleep or time.sleep
    attempt = 0
    rate_limited = 0
    while True:
        if limiter is not None:
            limiter.acquire(key, tokens)
        try:
            response = send()
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
                    return response
                rate_limited += 1
                wait = delay if delay is not None else backoff(rate_limited, base_delay, max_delay)
                logger.warning("%s: 429, all callers wait %.1fs", key, wait)
                limiter.block(key, wait)
                continue
            if attempt >= max_retries:
                return response
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
```

- [ ] **Step 4: Wire into `gemini_client.py`**

1. `GeminiModel`: add `GEMINI_3_5_FLASH_LITE = "gemini-3.5-flash-lite"` after `GEMINI_3_1_FLASH_LITE`.
2. Imports: `from .rate_limiter import estimate_audio_tokens, estimate_text_tokens` and `from .request_policy import send_with_limits`.
3. `__init__`: add parameter `rate_limiter=None` (last), docstring line `rate_limiter: Shared RateLimiter; when set it replaces the built-in throttle.`, and `self.rate_limiter = rate_limiter`.
4. Replace the body of `_post_with_retry` (keep its docstring first paragraph) with:

```python
    def _post_with_retry(self, url, payload, timeout, headers=None, limit_key=None, tokens=0,
                         max_retries=None):
        return send_with_limits(
            lambda: requests.post(url, json=payload, timeout=timeout, headers=headers or None),
            limiter=self.rate_limiter, key=limit_key or "gemini", tokens=tokens,
            max_retries=self.max_retries if max_retries is None else max_retries,
            base_delay=self.retry_base_delay, max_delay=self.max_retry_delay,
        )
```

5. `speech_to_text`: the throttle lines become

```python
        # Built-in spacing only when no shared limiter is managing this key.
        if self.rate_limiter is None:
            self._throttle()  # Minimum time between requests
            self._rate_limit()  # Requests per minute limit
```

and the call becomes `self._post_with_retry(request_url, payload, timeout=dynamic_timeout, headers=request_headers or None, limit_key=model_name, tokens=estimate_audio_tokens(duration_sec))`.

6. `generate_text`: replace the `attempt = 0 / while True` loop with

```python
        if self.rate_limiter is None:
            self._throttle()
            self._rate_limit()
        try:
            res = self._post_with_retry(request_url, payload, timeout=timeout or self.timeout,
                                        limit_key=model_name, tokens=estimate_text_tokens(prompt),
                                        max_retries=retries)
        except requests.exceptions.RequestException as exc:
            raise GeminiAPIError(f"request failed: {exc}") from exc
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_request_policy.py tests/test_gemini_client.py tests/test_gemini_base_url.py -q`
Expected: all pass (existing retry tests keep passing because `sleep` is resolved at call time, so `patch("time.sleep")` still applies).

- [ ] **Step 6: Commit** (when authorized): `feat: one retry policy shared by providers; Gemini uses the limiter`

---

### Task 3: Merge path and public `transcribe*` functions honour the limiter

**Files:**
- Modify: `src/multimodal_transcription_core/chunked_transcriber.py` (`_standard_transcribe`, `ChunkedMergeAPI.generate_text`, `transcribe_large`)
- Modify: `src/multimodal_transcription_core/transcriber.py` (`transcribe`)
- Modify: `src/multimodal_transcription_core/__init__.py` (exports)
- Test: `tests/test_limited_transcription.py`; existing chunked tests stay green

**Interfaces:**
- Consumes: `GeminiAPI(rate_limiter=...)`, `GeminiAPI.generate_text`, `RateLimitError`, `QuotaExhausted`
- Produces: `transcribe(audio_path, ..., api: GeminiAPI|None = None)`, `transcribe_large(audio_path, api: GeminiAPI|None = None)`; package exports `RateLimiter, Limits, QuotaExhausted, RateLimitTimeout, RateLimitError`

- [ ] **Step 1: Write the failing tests** — `tests/test_limited_transcription.py`

```python
"""Public transcribe functions use the caller's limited client end to end."""
from unittest.mock import MagicMock, patch

import pytest

import multimodal_transcription_core as core
from multimodal_transcription_core import chunked_transcriber, transcriber


def test_package_exports_limiter_types():
    for name in ("RateLimiter", "Limits", "QuotaExhausted", "RateLimitTimeout", "RateLimitError"):
        assert hasattr(core, name)


def test_transcribe_uses_the_given_client(tmp_path):
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    api = MagicMock()
    api.speech_to_text.return_value = {"text": "ok"}
    assert transcriber.transcribe(audio, api=api)["text"] == "ok"
    api.speech_to_text.assert_called_once()


def test_transcribe_lets_quota_errors_through_unwrapped(tmp_path):
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    api = MagicMock()
    api.speech_to_text.side_effect = core.QuotaExhausted("m", 0.0, "test")
    with pytest.raises(core.QuotaExhausted):
        transcriber.transcribe(audio, api=api)


def test_merge_goes_through_generate_text():
    api = MagicMock()
    api.generate_text.return_value = "merged"
    assert chunked_transcriber.ChunkedMergeAPI(api).generate_text("p") == "merged"
    assert api.generate_text.call_args.kwargs["model"] == chunked_transcriber.MERGE_MODEL


def test_merge_failure_returns_empty_but_quota_propagates():
    api = MagicMock()
    api.generate_text.side_effect = core.GeminiAPIError("boom")
    assert chunked_transcriber.ChunkedMergeAPI(api).generate_text("p") == ""
    api.generate_text.side_effect = core.QuotaExhausted("m", 0.0, "test")
    with pytest.raises(core.QuotaExhausted):
        chunked_transcriber.ChunkedMergeAPI(api).generate_text("p")


def test_quota_exhausted_aborts_chunked_transcription(tmp_path):
    audio = tmp_path / "long.m4a"
    audio.touch()
    chunks = [MagicMock(path=f"/tmp/c{i}.m4a", start_time=i * 1800, end_time=(i + 1) * 1800)
              for i in range(3)]
    api = MagicMock()
    with patch.object(chunked_transcriber, "get_duration", return_value=5400.0), \
         patch.object(chunked_transcriber, "split_audio", return_value=chunks), \
         patch.object(chunked_transcriber, "cleanup_chunks") as cleanup, \
         patch.object(chunked_transcriber, "_transcribe_chunk", side_effect=[
             {"text": "part one", "success": True},
             core.QuotaExhausted("m", 0.0, "test"),
             {"text": "part three", "success": True}]):
        with pytest.raises(core.QuotaExhausted):
            chunked_transcriber.transcribe_large(audio, api=api)
    cleanup.assert_called_once_with(chunks)


def test_short_file_passes_client_to_standard_path(tmp_path):
    audio = tmp_path / "short.m4a"
    audio.touch()
    api = MagicMock()
    with patch.object(chunked_transcriber, "get_duration", return_value=60.0), \
         patch.object(chunked_transcriber, "_standard_transcribe", return_value={"text": "t"}) as std:
        chunked_transcriber.transcribe_large(audio, api=api)
    assert std.call_args.kwargs["api"] is api
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_limited_transcription.py -q`
Expected: FAIL (`api` unexpected keyword, missing exports).

- [ ] **Step 3: Implement**

`transcriber.py`: add `api=None` as the last parameter of `transcribe` (docstring: `api: Configured GeminiAPI, e.g. with a shared RateLimiter; default builds one.`), import `from .rate_limiter import RateLimitError`, use `api = api or GeminiAPI()`, and add before `except Exception`:

```python
    except RateLimitError:
        raise
```

`chunked_transcriber.py`:

```python
_rate_limiter = _load_module("rate_limiter")
RateLimitError = _rate_limiter.RateLimitError
GeminiAPIError = _gemini_client.GeminiAPIError


def _standard_transcribe(audio_path: str, api: Optional[GeminiAPI] = None) -> Dict[str, Any]:
    transcriber = _load_module("transcriber")
    return transcriber.transcribe(audio_path, api=api)
```

`ChunkedMergeAPI.generate_text` body becomes:

```python
        try:
            return self.api.generate_text(prompt, model=self.model, temperature=0.0,
                                          timeout=120, retries=max_retries)
        except GeminiAPIError as exc:
            logger.error(f"Merge text generation failed: {exc}")
            return ""
```

`transcribe_large(audio_path, api=None)`: short path `return _standard_transcribe(str(path), api=api)`; `api = api or GeminiAPI()`; inside the chunk loop add before the generic `except Exception as e:`

```python
        except RateLimitError:
            # A quota stop must not leave silent holes in a merged transcript.
            cleanup_chunks(chunks)
            raise
```

`__init__.py`: add `from .rate_limiter import Limits, QuotaExhausted, RateLimiter, RateLimitError, RateLimitTimeout` and the five names to `__all__`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_limited_transcription.py tests/test_chunked_transcriber.py tests/test_gemini_base_url.py tests/test_transcriber.py -q`
Expected: all pass.

- [ ] **Step 5: Commit** (when authorized): `feat: transcribe and transcribe_large accept a limited client`

---

### Task 4: Groq client uses the same policy

**Files:**
- Modify: `src/multimodal_transcription_core/groq_client.py` (`__init__`, request block ~line 186-210)
- Test: `tests/test_groq_limits.py`

**Interfaces:**
- Consumes: `send_with_limits` (Task 2), `RateLimiter` (Task 1)
- Produces: `GroqAPI(throttle_rate=0.5, timeout=40, rate_limiter=None, max_retries=4, retry_base_delay=2.0, max_retry_delay=60.0)`

- [ ] **Step 1: Write the failing tests** — `tests/test_groq_limits.py`

```python
"""Groq waits on 429 instead of failing, and resends the whole file."""
from unittest.mock import MagicMock, patch

import requests

from multimodal_transcription_core import GroqAPI, Limits, RateLimiter


def resp(status, body, headers=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = body
    r.text = ""
    return r


def test_groq_retry_resends_full_audio(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF" + b"\0" * 64)
    sent = []

    def fake_post(url, files=None, **kw):
        sent.append(files["file"][1].read())
        if len(sent) == 1:
            return resp(429, {"error": {"message": "slow down"}}, {"retry-after": "3"})
        return resp(200, {"text": "привет"})

    with patch("requests.post", side_effect=fake_post), patch("time.sleep") as sleep, \
         patch("multimodal_transcription_core.audio_converter.AudioConverter.get_audio_duration",
               return_value=1.0):
        out = GroqAPI(retry_base_delay=0).speech_to_text(str(audio))
    assert out["text"] == "привет"
    assert sent[0] == sent[1] == audio.read_bytes()
    assert sleep.call_args_list[0].args[0] == 3.0


def test_groq_counts_against_the_limiter(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF" + b"\0" * 64)
    rl = RateLimiter({"whisper-large-v3": Limits(rpd=5)})
    with patch("requests.post", return_value=resp(200, {"text": "x"})), \
         patch("multimodal_transcription_core.audio_converter.AudioConverter.get_audio_duration",
               return_value=1.0):
        GroqAPI(rate_limiter=rl).speech_to_text(str(audio))
    assert rl.usage("whisper-large-v3")["today"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_groq_limits.py -q`
Expected: FAIL (`rate_limiter` unexpected keyword / 429 raised as STTAPIError).

- [ ] **Step 3: Implement**

`__init__` gains `rate_limiter=None, max_retries: int = 4, retry_base_delay: float = 2.0, max_retry_delay: float = 60.0` stored on `self`. Import `from .request_policy import send_with_limits`. Replace `self._throttle()` + the `files = {...}` line + `res = requests.post(...)` with:

```python
        # rate-limit: shared limiter when given, otherwise the built-in spacing
        if self.rate_limiter is None:
            self._throttle()
        ...
            res = send_with_limits(
                # A fresh buffer per attempt: a retry must resend the whole file.
                lambda: requests.post(
                    self.API_URL, headers=headers,
                    files={"file": (filename, io.BytesIO(audio_bytes), mime_type)},
                    data=data, timeout=self.timeout),
                limiter=self.rate_limiter, key=model.value, max_retries=self.max_retries,
                base_delay=self.retry_base_delay, max_delay=self.max_retry_delay,
            )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_groq_limits.py -q` → pass.

- [ ] **Step 5: Commit** (when authorized): `feat: Groq client waits on 429 and shares the limiter`

---

### Task 5: CLI reads limits from `transcription.ini` and shows usage

**Files:**
- Modify: `src/multimodal_transcription_core/runner.py` (`transcribe_with_model`, `main`)
- Modify: `tests/test_cli_runner.py` (stub accepts `limiter`; new tests)
- Modify: `transcription.ini`, `examples/quickstart/config.ini` (add `[limits]` and rpm/tpm/rpd)

**Interfaces:**
- Consumes: `RateLimiter.from_parser`, `RateLimiter.usage`, `GeminiAPI(rate_limiter=)`, `GroqAPI(rate_limiter=)`
- Produces: `transcribe_with_model(name, options, audio_path, language, limiter=None)`; `--list-models` lines end with `today N/RPD` for limited models

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cli_runner.py`; change the stub to `def fake_transcribe(model_name, options, audio_path, language, limiter=None):` and record `"limiter": limiter`)

```python
LIMITED = CONFIG + """
[limits]
state_file = {state}
"""


@pytest.fixture
def limited_config(tmp_path):
    env = tmp_path / "keys.env"
    env.write_text("GEMINI_API_KEY=stub\n")
    path = tmp_path / "limited.ini"
    text = textwrap.dedent(LIMITED).format(env_file=env, state=tmp_path / "state.sqlite")
    text = text.replace("[model.fast-cloud]\nkind = gemini",
                        "[model.fast-cloud]\nkind = gemini\nrpm = 15\nrpd = 500")
    path.write_text(text)
    return path


def test_runner_passes_a_limiter_built_from_config(limited_config, audio, stub_provider, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert runner.main([str(audio), "--config", str(limited_config)]) == 0
    limiter = stub_provider[0]["limiter"]
    assert limiter.limits["fast-cloud"].rpd == 500


def test_list_models_shows_daily_usage(capsys, limited_config, stub_provider, monkeypatch):
    runner.main(["--list-models", "--config", str(limited_config)])
    out = capsys.readouterr().out
    assert "today 0/500" in out
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_cli_runner.py -q` → the two new tests fail.

- [ ] **Step 3: Implement** in `runner.py`

```python
def transcribe_with_model(name, options, audio_path, language, limiter=None):
    ...
        return GeminiAPI(rate_limiter=limiter).speech_to_text(...)["text"]
    ...
        result = STTAPI(rate_limiter=limiter).speech_to_text(...)
```

In `main`, after `load_secrets(config)`: `from .rate_limiter import RateLimiter` then `limiter = RateLimiter.from_parser(config)`. In `--list-models`:

```python
            api_model = options.get("model", name)
            limits = limiter.limits.get(api_model)
            quota = ""
            if limits is not None and limits.rpd is not None:
                quota = f"  today {limiter.usage(api_model)['today']}/{limits.rpd}"
            print(f"{name:<28} {options.get('kind', '?'):<8} {state}{quota}")
```

Call: `text = transcribe_with_model(chosen, options, args.audio, language, limiter=limiter)`.

Config (`transcription.ini`, and the same shape in `examples/quickstart/config.ini`):

```ini
[limits]
# Local disk only - never /Volumes/VM-shared: SQLite locks are unreliable on SMB.
# Every app and run that names this file shares one queue.
state_file = ~/.local/state/transcription-core/ratelimit.sqlite
# Longest a request may wait for a slot before failing with RateLimitTimeout.
max_wait_seconds = 3600

[model.gemini-3.5-flash-lite]
kind = gemini
rpm = 15
tpm = 250K
rpd = 500

[model.gemini-3-flash-preview]
kind = gemini
rpm = 5
tpm = 250K
rpd = 20
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest -q` → full suite green.

- [ ] **Step 5: Commit** (when authorized): `feat: CLI applies configured limits and shows today's usage`

---

### Task 6: Integration guide, version, verification

**Files:**
- Create: `docs/integration.md` (Clip Smith, Exocortex, host install, config, errors, migration notes 0.1 → 0.2)
- Modify: `README.md` (short "Rate limits" section linking the guide), `pyproject.toml` (`version = "0.2.0"`), `docs/superpowers/specs/rate-limits.md` ("Done when": replace the benchmark-sample run with a one-request smoke on a generated clip)

- [ ] **Step 1: Write `docs/integration.md`** covering: one shared `transcription.ini` per machine at `~/.config/transcription/transcription.ini`; install into each app's venv (`pip install <path-to-checkout>` or a built wheel); the Python snippet (`RateLimiter.from_config(...)`, `GeminiAPI(rate_limiter=...)`, `transcribe(..., api=...)`); what `QuotaExhausted` / `RateLimitTimeout` mean and what the app should do; host vs VM (each machine has its own local state file, so quotas are only shared within a machine; split RPD between them or run transcription on one machine); Exocortex: keep its current implementation until its migration agent validates the installed core through the real app entry point (AGENTS.md).
- [ ] **Step 2: Offline verification**

```bash
.venv/bin/python -m pytest -q
rsync -a --exclude .venv --exclude .git --exclude '__pycache__' ./ "$SCRATCH/export/"
python3 -m venv "$SCRATCH/wheel-venv" && "$SCRATCH/wheel-venv/bin/pip" install build
"$SCRATCH/wheel-venv/bin/python" -m build --wheel "$SCRATCH/export" --outdir "$SCRATCH/dist"
python3 -m venv "$SCRATCH/smoke" && "$SCRATCH/smoke/bin/pip" install "$SCRATCH"/dist/*.whl
cd "$SCRATCH" && "$SCRATCH/smoke/bin/transcription-core" --list-models --config <repo>/transcription.ini
"$SCRATCH/smoke/bin/python" -c "import multimodal_transcription_core as c; print(c.RateLimiter)"
```

Expected: tests pass; wheel installs; `--list-models` prints `today N/500` for flash-lite.

- [ ] **Step 3: One live smoke (1 of 500 Flash Lite requests)** — generate a 3-second Russian clip with macOS `say`, convert to m4a with ffmpeg, run `transcription-core <clip> -m gemini-3.5-flash-lite --config transcription.ini`, then `--list-models` to show the request counted. Not a benchmark; benchmark files untouched.
- [ ] **Step 4: Inspect diff** (`git diff --stat`, `git status --short`) and confirm `benchmark/` unchanged.
