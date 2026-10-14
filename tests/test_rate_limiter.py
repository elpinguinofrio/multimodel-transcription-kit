"""RateLimiter: waits for a free slot, shares counts, stops on daily quota."""
import configparser
import multiprocessing
import sqlite3
import threading
import time
from datetime import datetime

import pytest

from multimodal_transcription_core.rate_limiter import (
    Limits, QuotaExhausted, RateLimiter, RateLimitTimeout, RequestTooLarge, QUOTA_DAY_TZ,
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


def test_request_bigger_than_tpm_is_refused_before_sending():
    clock = FakeClock()
    rl = limiter({"m": Limits(tpm=10_000)}, clock)
    with pytest.raises(RequestTooLarge, match="19700"):
        rl.acquire("m", tokens=19_700)
    assert rl.usage("m")["today"] == 0


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


def _contend(path, barrier):
    rl = RateLimiter({"m": Limits(rpm=3)}, state_path=path, max_wait=0)
    barrier.wait(30)
    try:
        rl.acquire("m")
    except RateLimitTimeout:
        raise SystemExit(3)


def test_processes_released_together_share_one_limit(tmp_path):
    path = str(tmp_path / "state" / "limits.sqlite")
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(6)
    procs = [ctx.Process(target=_contend, args=(path, barrier)) for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert sorted(p.exitcode for p in procs) == [0, 0, 0, 3, 3, 3]


def test_locked_database_ends_in_timeout_not_sqlite_error(tmp_path):
    path = tmp_path / "limits.sqlite"
    rl = RateLimiter({"m": Limits(rpm=3)}, state_path=path, max_wait=0.3)
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        with pytest.raises(RateLimitTimeout):
            rl.acquire("m")
        assert time.monotonic() - started < 1.0
        with pytest.raises(RateLimitTimeout):
            rl.block("m", 5)
    finally:
        holder.execute("ROLLBACK")
        holder.close()


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


def test_zero_limit_means_unavailable_not_a_crash():
    clock = FakeClock()
    rl = limiter({"m": Limits(rpm=0)}, clock)
    with pytest.raises(QuotaExhausted, match="0"):
        rl.acquire("m")


def test_from_parser_reads_another_section_prefix():
    parser = configparser.ConfigParser()
    parser.read_string("""
[provider.gemini]
enabled = true
kind = gemini
model = gemini-3-flash-preview
rpm = 5
rpd = 20

[model.ignored]
rpm = 1
""")
    rl = RateLimiter.from_parser(parser, section_prefix="provider.")
    assert rl.limits == {"gemini-3-flash-preview": Limits(rpm=5, rpd=20)}


def test_builtin_tier_ships_public_limits():
    from multimodal_transcription_core.rate_limiter import load_tier
    tier = load_tier("gemini-pro-tier")
    assert tier["gemini-3.5-flash-lite"] == Limits(rpm=15, tpm=250_000, rpd=500)
    assert tier["gemini-3-flash-preview"] == Limits(rpm=5, tpm=250_000, rpd=20)


def test_unknown_tier_names_the_available_ones():
    from multimodal_transcription_core.rate_limiter import load_tier
    with pytest.raises(FileNotFoundError, match="gemini-pro-tier"):
        load_tier("no-such-tier")


def test_config_picks_a_tier_and_overrides_single_fields():
    parser = configparser.ConfigParser()
    parser.read_string("""
[limits]
tiers = gemini-pro-tier

[model.flash-lite]
kind = gemini
model = gemini-3.5-flash-lite
rpd = 300
""")
    rl = RateLimiter.from_parser(parser, state_path=None)
    assert rl.limits["gemini-3.5-flash-lite"] == Limits(rpm=15, tpm=250_000, rpd=300)
    assert rl.limits["gemini-3-flash-preview"].rpd == 20


def test_tier_model_with_zero_limit_is_refused_clearly():
    parser = configparser.ConfigParser()
    parser.read_string("[limits]\ntiers = gemini-pro-tier\n")
    rl = RateLimiter.from_parser(parser, state_path=None)
    with pytest.raises(QuotaExhausted, match="limit is 0"):
        rl.acquire("gemini-3.1-pro-preview")
