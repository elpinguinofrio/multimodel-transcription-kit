"""Model choice, fallback and output records (owner decision, 2026-10-14).

Best = first model in the ranking that is reachable, has quota left and can supply
every explicitly requested feature. Without -m a failure moves down the ranking and
is recorded; with -m there is never a fallback.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

import multimodal_transcription_core as core
from multimodal_transcription_core import runner, selection
from multimodal_transcription_core.engine import ModelSpec

A = ModelSpec(name="a", kind="gemini", model="gemini-3.5-flash-lite")
B = ModelSpec(name="b", kind="groq", model="whisper-large-v3")
C = ModelSpec(name="c", kind="http", model="parakeet", endpoint="http://127.0.0.1:9/v1")


def ok(spec):
    return {"text": f"from {spec.name}", "warnings": [], "name": spec.name}


def test_limiter_reports_an_exhausted_day():
    limiter = core.RateLimiter({"m": core.Limits(rpd=1), "zero": core.Limits(rpd=0), "free": core.Limits()})
    assert limiter.exhausted("m") is None
    limiter.acquire("m")
    assert "1/1" in limiter.exhausted("m")
    assert "0" in limiter.exhausted("zero")
    assert limiter.exhausted("free") is None
    limiter.block_until_reset("free", "server said so")
    assert "server said so" in limiter.exhausted("free")


def test_first_available_model_in_ranking_is_used():
    calls = []
    result = selection.transcribe_best("x.wav", [A, B, C], probe=lambda s: s.name != "a",
                                       transcribe=lambda audio, spec, **kw: calls.append(spec.name) or ok(spec))
    assert calls == ["b"] and result["selection"]["chosen"] == "b"
    assert result["selection"]["attempts"] == [{"name": "a", "skipped": "not reachable"}]


def test_exhausted_quota_is_skipped():
    limiter = core.RateLimiter({A.model: core.Limits(rpd=0)})
    result = selection.transcribe_best("x.wav", [A, B], probe=lambda s: True, limiter=limiter,
                                       transcribe=lambda audio, spec, **kw: ok(spec))
    assert result["selection"]["chosen"] == "b"
    assert "quota" in result["selection"]["attempts"][0]["skipped"]


def test_models_that_cannot_supply_an_explicit_feature_are_skipped():
    result = selection.transcribe_best("x.wav", [B, A], probe=lambda s: True, speakers=True,
                                       transcribe=lambda audio, spec, **kw: ok(spec))
    assert result["selection"]["chosen"] == "a"
    assert "speaker" in result["selection"]["attempts"][0]["skipped"]


def test_a_failure_falls_back_down_the_ranking_and_is_recorded():
    def transcribe(audio, spec, **kw):
        if spec.name == "a":
            raise core.QuotaExhausted(spec.model, 0.0, "test")
        return ok(spec)
    result = selection.transcribe_best("x.wav", [A, B], probe=lambda s: True, transcribe=transcribe)
    assert result["selection"]["chosen"] == "b"
    assert result["selection"]["attempts"][0]["name"] == "a"
    assert "QuotaExhausted" in result["selection"]["attempts"][0]["error"]


def test_no_fallback_when_disallowed():
    def transcribe(audio, spec, **kw):
        raise core.ProviderError("API error 503")
    with pytest.raises(core.ProviderError):
        selection.transcribe_best("x.wav", [A, B], probe=lambda s: True, transcribe=transcribe,
                                  allow_fallback=False)


def test_all_failing_raises_with_every_attempt():
    def transcribe(audio, spec, **kw):
        raise core.ResponseError("blocked")
    with pytest.raises(core.NoModelAvailableError) as caught:
        selection.transcribe_best("x.wav", [A, B], probe=lambda s: True, transcribe=transcribe)
    assert [a["name"] for a in caught.value.attempts] == ["a", "b"]


@pytest.mark.parametrize("error", [FileNotFoundError("gone"), core.AudioPreparationError("no ffmpeg")])
def test_input_problems_are_not_retried_on_other_models(error):
    calls = []

    def transcribe(audio, spec, **kw):
        calls.append(spec.name)
        raise error
    with pytest.raises(type(error)):
        selection.transcribe_best("x.wav", [A, B], probe=lambda s: True, transcribe=transcribe)
    assert calls == ["a"]


def test_error_descriptions_name_retired_models_and_rejected_keys():
    gone = core.GeminiAPIError("API error 404: not found")
    gone.status = 404
    assert "retired" in selection.describe_error(gone)
    denied = core.GeminiAPIError("API error 403: denied")
    denied.status = 403
    assert "rejected" in selection.describe_error(denied)


# ---------------------------------------------------------------- remote verification

@contextmanager
def models_endpoint(status):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"name": "models/x"}).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("status,expected", [(200, "verified"), (404, "retired"), (403, "rejected"),
                                             (401, "rejected"), (500, "unverified")])
def test_verify_reports_retired_models_and_rejected_keys(monkeypatch, status, expected):
    with models_endpoint(status) as url:
        monkeypatch.setenv("GEMINI_BASE_URL", url)
        assert expected in selection.verify(A).reason
        monkeypatch.setattr(core.GroqAPI, "API_URL", f"{url}/openai/v1/audio/transcriptions")
        assert expected in selection.verify(B).reason


def test_verify_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    result = selection.verify(A)
    assert not result.ok and "GEMINI_API_KEY" in result.reason


@contextmanager
def http_models(status, body):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions", seen
    finally:
        server.shutdown()
        server.server_close()


def local(endpoint, model="parakeet-tdt-0.6b-v3"):
    return ModelSpec(name="parakeet", kind="http", model=model, endpoint=endpoint)


LISTED = {"data": [{"id": "parakeet-tdt-0.6b-v3", "installed": True},
                   {"id": "gigaam-v3-e2e-rnnt", "installed": False}]}


def test_verify_http_confirms_the_model_is_listed():
    with http_models(200, LISTED) as (endpoint, seen):
        result = selection.verify(local(endpoint))
    assert result.ok and "verified" in result.reason
    assert seen == ["/v1/models"]


def test_verify_http_model_missing_from_the_list_is_unavailable():
    with http_models(200, LISTED) as (endpoint, _):
        result = selection.verify(local(endpoint, "parakeet-renamed"))
    assert not result.ok
    assert "not listed" in result.reason and "parakeet-tdt-0.6b-v3" in result.reason


def test_verify_http_model_listed_but_not_installed_is_unavailable():
    with http_models(200, LISTED) as (endpoint, _):
        result = selection.verify(local(endpoint, "gigaam-v3-e2e-rnnt"))
    assert not result.ok and "not installed" in result.reason


@pytest.mark.parametrize("status,body,detail", [(404, b"not found", "HTTP 404"), (500, b"", "HTTP 500"),
                                                (200, b"<html>", "not understood"),
                                                (200, {"models": []}, "not understood")])
def test_verify_http_without_a_model_list_stays_reachable_but_unverified(status, body, detail):
    with http_models(status, body) as (endpoint, _):
        result = selection.verify(local(endpoint))
    assert result.ok and "unverified" in result.reason and detail in result.reason


def test_verify_http_endpoint_not_answering():
    with http_models(200, LISTED) as (endpoint, _):
        pass
    result = selection.verify(local(endpoint))
    assert not result.ok and "not answering" in result.reason


# ---------------------------------------------------------------- CLI

CONFIG = """
[models]
ranking = a, b

[model.a]
kind = gemini
model = gemini-3.5-flash-lite
temperature = 0.3

[model.b]
kind = groq
model = whisper-large-v3
"""


@pytest.fixture
def cli(tmp_path, monkeypatch):
    audio = tmp_path / "talk.wav"
    audio.write_bytes(b"x")
    config = tmp_path / "t.ini"
    config.write_text(CONFIG)
    monkeypatch.setattr(runner, "is_reachable", lambda name, options: True)
    calls = []

    def fake(name, options, audio_path, language, limiter=None, **kwargs):
        calls.append(name)
        if name == "a":
            raise core.ResponseError("blocked (finishReason=SAFETY)")
        return {"text": "hello", "warnings": ["b cannot supply speaker labels; continuing without them"],
                "name": name, "model": options["model"], "provider": options["kind"],
                "settings": {"language": language, "temperature": None},
                "features": {"speakers": False, "sounds": False, "timestamps": "major"},
                "audio": {"source": str(audio_path), "sha256": "ab" * 32, "converted": False},
                "chunks": 1}
    monkeypatch.setattr(runner, "transcribe_with_model", fake)
    return audio, config, calls


def test_cli_falls_back_and_records_it_in_the_document(cli, tmp_path):
    audio, config, calls = cli
    out = tmp_path / "out"
    out.mkdir()
    assert runner.main([str(audio), "--config", str(config), "-d", str(out)]) == 0
    assert calls == ["a", "b"]
    written = (out / "talk.b.md").read_text()
    lines = written.splitlines()
    assert lines[0] == f"src: {audio.resolve()}"
    assert "model: b (groq whisper-large-v3)" in lines
    assert any(l.startswith("fallback: a: ResponseError") for l in lines)
    assert any(l.startswith("settings: ") and '"temperature"' in l for l in lines)
    assert any(l.startswith("features: ") for l in lines)
    assert any(l.startswith("audio: ") and "sha256" in l for l in lines)
    assert any(l.startswith("warning: b cannot supply speaker labels") for l in lines)
    assert written.endswith("\n\nhello\n")


def test_cli_explicit_model_never_falls_back(cli, capsys):
    audio, config, calls = cli
    assert runner.main([str(audio), "--config", str(config), "-m", "a"]) == 1
    assert calls == ["a"]
    assert "ResponseError" in capsys.readouterr().err


def test_concurrent_runs_never_overwrite_each_other(tmp_path):
    audio = tmp_path / "talk.wav"
    audio.touch()
    out = tmp_path / "out"
    out.mkdir()
    barrier = threading.Barrier(8)
    written = []

    def write(n):
        barrier.wait()
        written.append(runner.write_transcript(out, audio, "m", f"body {n}\n"))
    threads = [threading.Thread(target=write, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(written)) == 8 and len(list(out.iterdir())) == 8
    assert sorted(p.read_text() for p in out.iterdir()) == sorted(f"body {n}\n" for n in range(8))


def test_existing_outputs_are_preserved(tmp_path):
    audio = tmp_path / "talk.wav"
    audio.touch()
    (tmp_path / "talk.m.md").write_text("old")
    assert runner.write_transcript(tmp_path, audio, "m", "new").name == "talk.m.(2).md"
    assert (tmp_path / "talk.m.md").read_text() == "old"


def test_list_models_reports_retired_models(tmp_path, monkeypatch, capsys):
    config = tmp_path / "t.ini"
    config.write_text("[models]\nranking = a\n[model.a]\nkind = gemini\nmodel = gemini-2-old\n")
    with models_endpoint(404) as url:
        monkeypatch.setenv("GEMINI_BASE_URL", url)
        assert runner.main(["--list-models", "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "a" in out and "retired" in out
