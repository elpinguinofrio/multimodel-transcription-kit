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
# Seconds SQLite waits for another process's lock before acquire() re-checks its deadline.
LOCK_TIMEOUT = 2.0
# Brief lock collisions are not "waiting for a slot": even max_wait=0 tolerates this much.
MIN_LOCK_WAIT = 0.5
LOCK_RETRY_DELAY = 0.05

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


class RequestTooLarge(RateLimitError):
    """One request needs more tokens than the model allows per minute."""


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
            db = sqlite3.connect(self._path, timeout=LOCK_TIMEOUT, isolation_level=None)
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
    def from_parser(cls, parser: configparser.ConfigParser, section_prefix: str = "model.",
                    **kwargs) -> "RateLimiter":
        """[limits] state_file/max_wait_seconds and rpm/tpm/rpd in each [<prefix><name>].

        The quota key is the section's `model` option, else the name after the prefix.
        """
        limits = {}
        for section in parser.sections():
            if not section.startswith(section_prefix):
                continue
            options = parser[section]
            found = Limits(parse_count(options.get("rpm")), parse_count(options.get("tpm")),
                           parse_count(options.get("rpd")))
            if found != Limits():
                limits[options.get("model", section[len(section_prefix):])] = found
        kwargs.setdefault("state_path", parser.get("limits", "state_file", fallback=None))
        kwargs.setdefault("max_wait", parser.getfloat("limits", "max_wait_seconds", fallback=3600.0))
        return cls(limits, **kwargs)

    # ───────────────────────── storage ──────────────────────────

    @contextmanager
    def _transaction(self, write=True, lock_timeout=LOCK_TIMEOUT):
        with self._lock:
            db = self._shared or sqlite3.connect(self._path, timeout=max(lock_timeout, 0.0),
                                                 isolation_level=None)
            try:
                db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
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
        limits = self.limits.get(key)
        if limits is not None and 0 in (limits.rpm, limits.tpm, limits.rpd):
            raise QuotaExhausted(key, next_quota_reset(self._clock()),
                                 "limit is 0 for this model on this API tier")
        if limits is not None and limits.tpm is not None and tokens > limits.tpm:
            raise RequestTooLarge(
                f"{key}: one request needs ~{tokens} tokens but the limit is {limits.tpm} per minute; "
                "split the audio or use a model with a larger token limit")
        deadline = self._clock() + self.max_wait
        while True:
            try:
                remaining = deadline - self._clock()
                lock_timeout = min(LOCK_TIMEOUT, max(remaining, MIN_LOCK_WAIT))
                with self._transaction(lock_timeout=lock_timeout) as db:
                    wait = self._reserve(db, key, tokens)
            except sqlite3.OperationalError as exc:
                # Another process holds the state file; keep trying until our own deadline.
                if "locked" not in str(exc) or self._clock() >= deadline - MIN_LOCK_WAIT:
                    raise RateLimitTimeout(f"{key}: limiter state stayed locked ({exc})") from exc
                time.sleep(LOCK_RETRY_DELAY)
                continue
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
        with self._transaction(write=False) as db:
            (today,) = db.execute("SELECT COUNT(*) FROM requests WHERE key=? AND ts>=?",
                                  (key, quota_day_start(now))).fetchone()
            minute, tokens = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM requests WHERE key=? AND ts>?",
                (key, now - WINDOW_SECONDS)).fetchone()
        return {"today": today, "last_minute": minute, "tokens_last_minute": tokens}

    # ───────────────────────── internals ──────────────────────────

    def _set_block(self, key, until, daily, reason):
        try:
            with self._transaction() as db:
                row = db.execute("SELECT until FROM blocks WHERE key=?", (key,)).fetchone()
                if row and row[0] >= until:
                    return
                db.execute("INSERT OR REPLACE INTO blocks (key, until, daily, reason) VALUES (?, ?, ?, ?)",
                           (key, until, int(daily), reason))
        except sqlite3.OperationalError as exc:
            raise RateLimitTimeout(f"{key}: limiter state stayed locked ({exc})") from exc

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
