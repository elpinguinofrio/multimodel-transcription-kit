"""transcription.toml providers: accounts by type, models best-first, paid routes last.

Providers are the synthetic local gateway from test_engine and a fake agy script.
BLIND SPOT: no real provider; live runs are reported separately.
"""
import configparser
import json
import stat
import tempfile
from pathlib import Path

import pytest

from multimodel_transcription_kit import engine, runner, selection
from multimodel_transcription_kit.engine import ModelSpec
from multimodel_transcription_kit.errors import ProviderError
from multimodel_transcription_kit.rate_limiter import RateLimiter
from multimodel_transcription_kit.toml_config import ConfigError, convert_ini, load
from test_engine import gateway, tone

ROOT = Path(__file__).resolve().parents[1]

CONFIG = """
[[aistudio]]
name = "free-account"
key_env = "FREE_KEY"

[[aistudio]]
name = "proxy"
key_env = "PROXY_KEY"
base_url = "http://127.0.0.1:9"
costs_money = true

[[antigravity]]
name = "agy"

[[openai]]
name = "local"
base_url = "{local}"

[[model]]
name = "best"
providers = ["agy:best-low", "free-account", "proxy"]

[[model]]
name = "second"
providers = ["free-account", "proxy"]

[[model]]
name = "timer"
providers = ["local"]
word_times = true
"""


def write(tmp_path, text=CONFIG, **values):
    path = tmp_path / "t.toml"
    path.write_text(text.format(local=values.get("local", "http://127.0.0.1:9/v1")), encoding="utf-8")
    return path


def ranking(parser, key="ranking"):
    return [n.strip() for n in parser["models"][key].split(",")]


def test_every_free_route_comes_before_any_paid_route(tmp_path):
    parser = load(write(tmp_path))
    assert ranking(parser) == ["best@agy", "best@free-account", "second@free-account", "timer@local",
                               "best@proxy", "second@proxy"]
    assert ranking(parser, "timing_ranking") == ["timer@local"]
    assert parser["model_routes"]["best"] == "best@agy, best@free-account, best@proxy"


def test_a_route_carries_its_account_address_key_and_model_id(tmp_path):
    parser = load(write(tmp_path))
    agy = ModelSpec.from_options("best@agy", dict(parser["model.best@agy"]))
    proxy = ModelSpec.from_options("best@proxy", dict(parser["model.best@proxy"]))
    local = ModelSpec.from_options("timer@local", dict(parser["model.timer@local"]))
    assert (agy.kind, agy.model, agy.account) == ("agy", "best-low", "agy")
    assert (proxy.kind, proxy.base_url, proxy.requires_env, proxy.costs_money) == (
        "gemini", "http://127.0.0.1:9", "PROXY_KEY", True)
    assert (local.kind, local.base_url, local.requires_env, local.costs_money) == (
        "openai", "http://127.0.0.1:9/v1", None, False)
    assert proxy.limit_key == "proxy/best" and agy.limit_key == "agy/best-low"


@pytest.mark.parametrize("text,message", [
    ('[[model]]\nname = "m"\nproviders = ["nobody"]\n', "no provider named 'nobody'"),
    ('[[aistudio]]\nname = "a"\nkey = "secret"\n[[model]]\nname = "m"\nproviders = ["a"]\n', "unknown setting"),
    ('[[aistudio]]\nname = "a"\n[[openai]]\nname = "a"\n[[model]]\nname = "m"\nproviders = ["a"]\n', "two providers"),
    ('[aistudio]\nname = "a"\n', "must be written [[aistudio]]"),
    ('[providers]\nx = 1\n', "unknown section"),
    ('[[antigravity]]\nname = "g"\nbase_url = "http://x"\n[[model]]\nname = "m"\nproviders = ["g"]\n',
     "takes no base_url"),
    ('[[aistudio]]\nname = "a"\n', "at least one [[model]]"),
])
def test_config_mistakes_are_named(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load(write(tmp_path, text))


def test_an_exhausted_account_does_not_block_the_same_model_on_another_account(tmp_path):
    limiter = RateLimiter({})
    limiter.block_until_reset("free-account/second", "server reported the daily quota used up")
    specs = [ModelSpec(name=f"second@{a}", kind="gemini", model="second", account=a) for a in ("free-account", "proxy")]
    calls = []

    def transcribe(audio, spec, **kw):
        calls.append(spec.name)
        return {"text": "ok", "name": spec.name}
    result = selection.transcribe_best("a.wav", specs, probe=lambda s: True, transcribe=transcribe, limiter=limiter)
    assert calls == ["second@proxy"]
    assert result["selection"]["attempts"][0]["skipped"].startswith("quota:")


def test_scoped_limiter_keeps_separate_counts_and_shares_model_limits():
    from multimodel_transcription_kit.rate_limiter import Limits
    limiter = RateLimiter({"m": Limits(rpd=1)})
    limiter.scoped("a").acquire("m")
    assert limiter.exhausted("a/m") and not limiter.exhausted("b/m")
    assert limiter.scoped(None) is limiter


def test_gemini_client_uses_the_accounts_key_variable_and_address(monkeypatch):
    from multimodel_transcription_kit.gemini_client import GeminiAPI
    monkeypatch.setenv("PROXY_KEY", "proxy-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "default-secret")
    api = GeminiAPI(key_env="PROXY_KEY", base_url="http://127.0.0.1:8300/")
    assert api.api_key == "proxy-secret" and api.API_URL.startswith("http://127.0.0.1:8300/v1beta/")
    assert GeminiAPI().api_key == "default-secret"


def test_engine_sends_an_aistudio_route_to_its_own_address_with_its_key(tmp_path, monkeypatch):
    monkeypatch.setenv("PROXY_KEY", "proxy-secret")
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    audio = tone(tmp_path / "a.wav", 2)
    with gateway("gemini") as (url, seen):
        spec = ModelSpec(name="m@proxy", kind="gemini", model="gemini-x", account="proxy", base_url=url,
                         requires_env="PROXY_KEY")
        result = engine.transcribe_file(audio, spec, limiter=RateLimiter({}), enhance=False)
    assert result["text"].strip() == "hello world"
    assert "key=proxy-secret" in seen[0]["path"] and seen[0]["path"].startswith("/v1beta/models/gemini-x")


@pytest.mark.parametrize("key", [None, "LOCAL_KEY"])
def test_openai_route_sends_a_key_only_when_it_has_one(tmp_path, monkeypatch, key):
    monkeypatch.setenv("LOCAL_KEY", "local-secret")
    audio = tone(tmp_path / "a.wav", 2)
    with gateway("groq") as (url, seen):
        spec = ModelSpec(name="w@local", kind="openai", model="whisper-x", account="local", base_url=f"{url}/v1",
                         requires_env=key)
        result = engine.transcribe_file(audio, spec, limiter=RateLimiter({}), enhance=False, timestamps="phrase")
    assert seen[0]["path"] == "/v1/audio/transcriptions"
    assert seen[0]["headers"].get("Authorization") == ("Bearer local-secret" if key else None)
    assert result["segments"] and result["provider"] == "openai"


def fake_agy(tmp_path, body):
    script = tmp_path / "agy"
    script.write_text("#!/bin/sh\n" + body + "\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_antigravity_route_returns_the_transcript_agy_prints(tmp_path, monkeypatch):
    log = tmp_path / "args.txt"
    monkeypatch.setenv("AGY_BIN", str(fake_agy(tmp_path, f'printf "%s\\n" "$@" > "{log}"; echo "привет мир"')))
    audio = tone(tmp_path / "a.wav", 2)
    spec = ModelSpec(name="best@agy", kind="agy", model="best-low", account="agy")
    result = engine.transcribe_file(audio, spec, terms=["Claude"], limiter=RateLimiter({}), enhance=False)
    args = log.read_text().splitlines()
    assert result["text"].strip() == "привет мир"
    assert args[args.index("--model") + 1] == "best-low" and args[args.index("--mode") + 1] == "plan"
    instruction = args[args.index("-p") + 1]
    sent = instruction.split("Read the audio file ", 1)[1].split(" with your", 1)[0]   # the source or a converted copy
    assert Path(sent).parent == Path(args[args.index("--add-dir") + 1])
    assert "Claude" in " ".join(args)


@pytest.mark.parametrize("body,message", [
    ("echo 'quota exhausted' >&2; exit 3", "exited 3"),
    ("exit 0", "no transcript"),
    ("echo CANNOT_HEAR_AUDIO", "could not hear"),
])
def test_antigravity_failures_raise_provider_errors(tmp_path, monkeypatch, body, message):
    monkeypatch.setenv("AGY_BIN", str(fake_agy(tmp_path, body)))
    from multimodel_transcription_kit.antigravity_client import transcribe
    with pytest.raises(ProviderError, match=message):
        transcribe(tone(tmp_path / "a.wav", 1), "best-low")


def test_cli_falls_back_from_a_failing_antigravity_route_to_the_next_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_BIN", str(fake_agy(tmp_path, "exit 1")))
    audio = tone(tmp_path / "a.wav", 2)
    out = tmp_path / "out"
    out.mkdir()
    text = ('[[antigravity]]\nname = "agy"\n\n[[openai]]\nname = "local"\nbase_url = "{local}"\n\n'
            '[[model]]\nname = "best"\nproviders = ["agy", "local"]\n')
    with gateway("groq", texts=["from the local server"]) as (url, _):
        config = write(tmp_path, text, local=f"{url}/v1")
        assert runner.main([str(audio), "--config", str(config), "-d", str(out), "--receipt"]) == 0
    receipt = json.loads(next(out.glob("*.receipt.json")).read_text())
    assert receipt["spec"]["account"] == "local" and receipt["spec"]["model"] == "best"
    assert receipt["result"]["selection"]["attempts"][0]["name"] == "best@agy"
    assert "from the local server" in receipt["result"]["text"]


def test_cli_model_flag_tries_that_models_providers_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv("FREE_KEY", "k")
    monkeypatch.setenv("PROXY_KEY", "k")
    calls = []

    def transcribe_with_model(name, options, audio_path, language, **kw):
        calls.append(name)
        if name == "second@free-account":
            raise ProviderError("quota")
        return {"text": "x", "name": name, "warnings": []}
    monkeypatch.setattr(runner, "transcribe_with_model", transcribe_with_model)
    monkeypatch.setattr(runner, "is_reachable", lambda name, options: True)
    assert runner.main([str(tone(tmp_path / "a.wav", 1)), "--config", str(write(tmp_path)), "-m", "second"]) == 0
    assert calls == ["second@free-account", "second@proxy"]


def test_unreachable_address_and_missing_agy_are_not_reachable(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_BIN", str(tmp_path / "missing"))
    parser = load(write(tmp_path))
    assert not runner.is_reachable("best@agy", dict(parser["model.best@agy"]))
    assert not runner.is_reachable("timer@local", dict(parser["model.timer@local"]))   # port 9: nothing listens


def test_ini_configs_convert_to_the_same_models_and_word_time_order():
    old = configparser.ConfigParser(interpolation=None)
    old.read(ROOT / "transcription.example.ini", encoding="utf-8")
    converted = convert_ini(old)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "converted.toml"
        target.write_text(converted, encoding="utf-8")
        new = load(target)
    old_ranking = [n.strip() for n in old["models"]["ranking"].split(",") if n.strip()]
    old_timing = [n.strip() for n in old["models"]["timing_ranking"].split(",") if n.strip()]
    timing_only = [n for n in old_timing if n not in old_ranking]
    # Timing-only models become text fallbacks too (the file says so); the old text order is kept.
    new_ranking = [r.split("@")[0] for r in ranking(new)]
    assert [n for n in new_ranking if n in old_ranking] == old_ranking and set(new_ranking) - set(old_ranking) == set(timing_only)
    assert all(name in converted.split("# NOTE:")[1] for name in timing_only)
    assert [r.split("@")[0] for r in ranking(new, "timing_ranking")] == old_timing
    assert new["vocabulary"]["corrections"] == "\n".join(
        f"{h} -> {r}" for h, r in __import__("multimodel_transcription_kit.vocabulary", fromlist=["x"])
        .parse_corrections(old["vocabulary"]["corrections"]))


def test_example_toml_loads_and_puts_the_paid_proxy_last():
    parser = load(ROOT / "transcription.example.toml")
    assert ranking(parser)[0] == "gemini-3.8-flash@my-agy"
    assert all(r.endswith("@my-proxy") for r in ranking(parser)[-3:])
    assert ranking(parser, "timing_ranking") == ["parakeet-tdt-0.6b-v3@my-local-parakeet",
                                                 "whisper-large-v3-turbo@my-groq"]


def test_openai_route_check_finds_a_model_in_the_servers_model_list(monkeypatch):
    # Local servers list /v1/models but do not answer /v1/models/<id> (Groq answers both).
    import requests

    class Response:
        def __init__(self, status, data=None):
            self.status_code, self._data = status, data

        def json(self):
            return self._data
    seen = []

    def get(url, headers=None, **kw):
        seen.append(headers or {})
        if url.endswith("/v1/models"):
            return Response(200, {"data": [{"id": "parakeet-tdt-0.6b-v3"}]})
        return Response(404)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr("socket.create_connection", lambda *a, **kw: type("S", (), {"close": lambda self: None})())
    spec = ModelSpec(name="p@local", kind="openai", model="parakeet-tdt-0.6b-v3", account="local",
                     base_url="http://127.0.0.1:8765/v1")
    assert selection.verify(spec) == selection.Availability(True, "verified: server lists the model")
    monkeypatch.setenv("LOCAL_KEY", "s")
    keyed = ModelSpec(name="w@g", kind="openai", model="parakeet-tdt-0.6b-v3", account="g",
                      base_url="http://127.0.0.1:8765/v1", requires_env="LOCAL_KEY")
    assert selection.verify(keyed).ok and seen[-1].get("Authorization") == "Bearer s"


@pytest.mark.parametrize("printed", ["Не попадайтесь.\n\n[Nov 8 : 4:20pm]\n", "Не попадайтесь.\n[16:20]\n"])
def test_antigravity_drops_the_clock_line_agy_appends(tmp_path, monkeypatch, printed):
    monkeypatch.setenv("AGY_BIN", str(fake_agy(tmp_path, f"printf '{printed}'")))
    from multimodel_transcription_kit.antigravity_client import transcribe
    assert transcribe(tone(tmp_path / "a.wav", 1), "best-low")["text"] == "Не попадайтесь."


def test_antigravity_keeps_a_bracket_inside_the_last_spoken_line(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_BIN", str(fake_agy(tmp_path, "printf 'Встречаемся [в 4:20]\\n'")))
    from multimodel_transcription_kit.antigravity_client import transcribe
    assert transcribe(tone(tmp_path / "a.wav", 1), "best-low")["text"] == "Встречаемся [в 4:20]"
