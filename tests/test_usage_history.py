"""Usage history survives processes and includes every physical provider attempt."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import multimodal_transcription_core as core


def log_at(path):
    cls = getattr(core, "UsageLog", None)
    assert cls is not None, "Core must expose persistent UsageLog"
    return cls(path)


@contextmanager
def gateway(responses):
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            observed.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            status, data = responses[min(len(observed) - 1, len(responses) - 1)]
            encoded = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", observed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


GEMINI = {
    "candidates": [{"content": {"parts": [{"text": "A complete sentence."}]}}],
    "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 4,
                      "thoughtsTokenCount": 3, "totalTokenCount": 19,
                      "cachedContentTokenCount": 2},
    "modelVersion": "gemini-3.6-flash",
}


def test_gemini_retries_and_merges_are_persistent_without_private_payload(tmp_path, monkeypatch):
    db = tmp_path / "usage.sqlite"
    history = log_at(db)
    limiter = core.RateLimiter({"gemini-3.6-flash": core.Limits(rpd=20)})
    with gateway([(429, {"error": {"message": "minute quota"}}), (200, GEMINI)]) as (url, observed):
        monkeypatch.setenv("GEMINI_BASE_URL", url)
        api = core.GeminiAPI(rate_limiter=limiter, usage_log=history)
        assert api.generate_text("PRIVATE PROMPT") == "A complete sentence."
    rows = log_at(db).requests()
    assert len(rows) == 2
    assert [r["http_status"] for r in rows] == [429, 200]
    assert [r["attempt"] for r in rows] == [1, 2]
    assert len({r["request_id"] for r in rows}) == 1
    assert rows[1]["input_tokens"] == 12
    assert rows[1]["output_tokens"] == 4
    assert rows[1]["total_tokens"] == 19
    assert rows[1]["thought_tokens"] == 3
    assert rows[1]["cached_tokens"] == 2
    assert rows[1]["response_model"] == "gemini-3.6-flash"
    assert all(r["provider"] == "gemini" and r["operation"] == "generation" for r in rows)
    assert limiter.usage("gemini-3.6-flash")["today"] == 2
    assert all("/models/gemini-3.6-flash:" in p for p, _ in observed)
    raw = db.read_bytes()
    assert b"PRIVATE PROMPT" not in raw and b"unit-test-key" not in raw
    assert b"A complete sentence" not in raw


def test_default_large_transcription_records_chunks_and_merge_under_36(tmp_path, monkeypatch):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"synthetic")
    chunks = [SimpleNamespace(path=str(audio), start_time=i * 1800, end_time=(i + 1) * 1800)
              for i in range(2)]
    from multimodal_transcription_core import chunked_transcriber as ct
    history = log_at(tmp_path / "usage.sqlite")
    limiter = core.RateLimiter({"gemini-3.6-flash": core.Limits(rpd=20)})
    with gateway([(200, GEMINI)]) as (url, _):
        monkeypatch.setenv("GEMINI_BASE_URL", url)
        api = core.GeminiAPI(rate_limiter=limiter, usage_log=history)
        with patch.object(ct, "get_duration", return_value=3600), \
             patch.object(ct, "split_audio", return_value=chunks), \
             patch.object(ct, "cleanup_chunks"), \
             patch.object(api, "_get_audio_duration", return_value=1800):
            result = core.transcribe_large(audio, api=api)
    assert result["model"] == "gemini-3.6-flash"
    assert [r["operation"] for r in history.requests()] == ["transcription", "transcription", "generation"]
    assert limiter.usage("gemini-3.6-flash")["today"] == 3
    summary = history.summary()[0]
    assert (summary["requests"], summary["input_tokens"], summary["output_tokens"]) == (3, 36, 12)


def test_groq_records_reported_usage(tmp_path):
    history = log_at(tmp_path / "usage.sqlite")
    with gateway([(200, {"text": "hello", "usage": {"prompt_tokens": 7, "completion_tokens": 2,
                                                     "total_tokens": 9}})]) as (url, _):
        api = core.GroqAPI(throttle_rate=0, usage_log=history)
        api.API_URL = url + "/audio/transcriptions"
        assert api.speech_to_text(b"synthetic")["text"] == "hello"
    row = history.requests()[0]
    assert (row["provider"], row["model"], row["input_tokens"], row["output_tokens"]) == (
        "groq", "whisper-large-v3", 7, 2)


def test_http_cli_and_usage_query_work_across_processes(tmp_path):
    db = tmp_path / "usage.sqlite"
    audio = tmp_path / "synthetic.wav"
    audio.write_bytes(b"synthetic")
    # No token usage is returned: unknown must survive in rows and totals.
    with gateway([(200, {"text": "hello"})]) as (url, _):
        command = [sys.executable, "-m", "multimodal_transcription_core", "http", "--audio", str(audio),
                   "--endpoint", url, "--output", str(tmp_path / "output"), "--usage-file", str(db)]
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    command = [sys.executable, "-m", "multimodal_transcription_core", "usage", "--db", str(db)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)[0]
    assert summary["provider"] == "http" and summary["requests"] == 1
    assert summary["input_tokens"] is None and summary["output_tokens"] is None
    assert summary["unknown_input_requests"] == 1
    assert summary["unknown_output_requests"] == 1


def test_http_errors_are_recorded_without_server_diagnostics(tmp_path):
    from urllib.error import HTTPError
    history = log_at(tmp_path / "usage.sqlite")
    audio = tmp_path / "synthetic.wav"
    audio.write_bytes(b"synthetic")
    with gateway([(403, {"error": "PRIVATE SECRET"})]) as (url, _):
        with pytest.raises(HTTPError):
            core.transcribe_http(audio, url, tmp_path / "output", usage_log=history)
    row = history.requests()[0]
    assert row["status"] == "error" and row["http_status"] == 403
    assert b"PRIVATE SECRET" not in history.path.read_bytes()


def test_transport_failure_is_recorded_before_exception_returns(tmp_path, monkeypatch):
    import requests
    history = log_at(tmp_path / "usage.sqlite")
    api = core.GeminiAPI(usage_log=history, max_retries=0, throttle_rate=0)
    def fail(*args, **kwargs):
        raise requests.exceptions.ConnectionError("PRIVATE KEY AND URL")
    monkeypatch.setattr(requests, "post", fail)
    with pytest.raises(core.GeminiAPIError):
        api.generate_text("private", retries=0)
    row = history.requests()[0]
    assert row["status"] == "error" and row["error_type"] == "ConnectionError"
    assert row["http_status"] is None
    assert b"PRIVATE KEY" not in history.path.read_bytes()


def test_history_queries_filter_page_and_preserve_unknowns(tmp_path):
    history = log_at(tmp_path / "usage.sqlite")
    with history.track("http", "local", "transcription") as event:
        event.update(http_status=200, response={"usage": {"prompt_tokens": 0, "completion_tokens": 0}})
    with history.track("http", "local", "transcription"):
        pass
    with history.track("groq", "other", "transcription"):
        pass
    rows = history.requests(provider="http", model="local", limit=1, offset=1)
    assert len(rows) == 1 and rows[0]["input_tokens"] is None
    summary = history.summary(provider="http")[0]
    assert summary["requests"] == 2 and summary["input_tokens"] == 0
    assert summary["unknown_input_requests"] == 1
    assert history.requests(since="2099-01-01T00:00:00Z") == []
    assert history.requests(until="2000-01-01T00:00:00Z") == []
    assert history.requests(model="local' OR 1=1 --") == []
    with pytest.raises(ValueError):
        history.requests(limit=-1)


@pytest.mark.parametrize("usage", [None, [], {"prompt_tokens": "12", "completion_tokens": -1},
                                   {"prompt_tokens": True, "completion_tokens": 1.5}])
def test_invalid_usage_stays_unknown(tmp_path, usage):
    history = log_at(tmp_path / "usage.sqlite")
    with history.track("http", "local", "transcription") as event:
        event["response"] = {"usage": usage}
    row = history.requests()[0]
    assert row["input_tokens"] is None and row["output_tokens"] is None


def test_concurrent_processes_append_without_losing_history(tmp_path):
    db = tmp_path / "usage.sqlite"
    code = ("from multimodal_transcription_core import UsageLog; import sys; "
            "log=UsageLog(sys.argv[1]); "
            "\nfor i in range(5):\n with log.track('http', 'local', 'transcription'): pass\n")
    processes = [subprocess.Popen([sys.executable, "-c", code, str(db)], cwd=tmp_path,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(3)]
    for process in processes:
        _, error = process.communicate(timeout=15)
        assert process.returncode == 0, error
    history = log_at(db)
    assert history.summary()[0]["requests"] == 15


def test_groq_retry_logs_both_attempts_without_limits(tmp_path):
    history = log_at(tmp_path / "usage.sqlite")
    with gateway([(429, {"error": "minute quota"}), (200, {"text": "hello"})]) as (url, _):
        api = core.GroqAPI(throttle_rate=0, usage_log=history)
        api.API_URL = url
        assert api.speech_to_text(b"synthetic")["text"] == "hello"
    summary = history.summary()[0]
    assert (summary["requests"], summary["calls"], summary["retries"], summary["failed"]) == (2, 1, 1, 1)
    assert summary["input_tokens"] is None


def test_history_accepts_concurrent_local_writers(tmp_path):
    db = tmp_path / "usage.sqlite"
    history = log_at(db)
    def record(i):
        with log_at(db).track("http", "local", "transcription"):
            pass
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record, range(24)))
    assert history.summary()[0]["requests"] == 24
    assert len({r["request_id"] for r in history.requests()}) == 24


def test_model_discovery_request_is_logged(tmp_path, monkeypatch):
    import requests
    history = log_at(tmp_path / "usage.sqlite")
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({"models": [{"name": "models/gemini-3.6-flash",
                                              "supportedGenerationMethods": ["generateContent"]}]}).encode()
    monkeypatch.setattr(requests, "get", lambda *a, **kw: response)
    assert core.GeminiAPI(usage_log=history).list_available_models() == ["gemini-3.6-flash"]
    row = history.requests()[0]
    assert row["operation"] == "models" and row["http_status"] == 200


@pytest.mark.parametrize("kind,model", [("gemini", "gemini-3.6-flash"),
                                        ("groq", "whisper-large-v3"), ("http", "local")])
def test_runner_enables_history_from_config_for_each_provider(tmp_path, monkeypatch, capsys, kind, model):
    from multimodal_transcription_core import runner
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"synthetic")
    db = tmp_path / "usage.sqlite"
    data = {**GEMINI, "text": "A complete sentence.",
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 19}}
    with gateway([(200, data)]) as (url, _):
        monkeypatch.setenv("GEMINI_BASE_URL", url)
        monkeypatch.setattr(core.GroqAPI, "API_URL", url)
        cfg = tmp_path / "config.ini"
        cfg.write_text(f"[usage]\nfile = {db}\n[models]\nranking = chosen\n"
                       f"[model.chosen]\nkind = {kind}\nmodel = {model}\nendpoint = {url}\n")
        assert runner.main([str(audio), "--config", str(cfg)]) == 0
    assert capsys.readouterr().out == "A complete sentence.\n"
    row = log_at(db).requests()[0]
    assert row["provider"] == kind and row["model"] == model and row["input_tokens"] == 12


def test_live_failure_is_recorded(tmp_path, monkeypatch):
    from test_gemini_live import gateway as live_gateway
    import wave
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x01\x00" * 3680)
    history = log_at(tmp_path / "usage.sqlite")
    with live_gateway(mode="setup_error") as (url, _):
        monkeypatch.setenv("GEMINI_BASE_URL", url)
        with pytest.raises(core.GeminiAPIError):
            core.transcribe_live(audio, "gemini-3.5-transcribe-live",
                                 api=core.GeminiAPI(timeout=2, usage_log=history))
    row = history.requests()[0]
    assert row["operation"] == "live" and row["status"] == "error"
    assert row["input_tokens"] is None


def test_query_never_creates_missing_database(tmp_path):
    db = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        log_at(db).summary()
    assert not db.exists()


def test_live_usage_is_logged(tmp_path, monkeypatch):
    from test_gemini_live import gateway as live_gateway
    import wave
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x01\x00" * 3680)
    history = log_at(tmp_path / "usage.sqlite")
    with live_gateway() as (url, _):
        monkeypatch.setenv("GEMINI_BASE_URL", url)
        result = core.transcribe_live(audio, "gemini-3.5-transcribe-live",
                                     api=core.GeminiAPI(timeout=2, usage_log=history))
    assert result["text"] == "Hello world"
    row = history.requests()[0]
    assert (row["provider"], row["operation"], row["input_tokens"], row["output_tokens"]) == (
        "gemini", "live", 7, 3)
