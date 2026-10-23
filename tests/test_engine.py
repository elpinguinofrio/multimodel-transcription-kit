"""One transcription path for the CLI and the library, for every provider kind.

Providers are a synthetic local HTTP server; ffmpeg/ffprobe are real (tests that
need them skip without them). BLIND SPOT: no real provider is contacted.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import shutil
import struct
import threading
import wave

import pytest

import multimodel_transcription_kit as core
from multimodel_transcription_kit import audio_chunker, engine, runner
from multimodel_transcription_kit.engine import ModelSpec

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="requires ffmpeg and ffprobe")

PARTS = ["alpha bravo charlie delta echo foxtrot golf",
         "delta echo foxtrot golf hotel india juliet kilo",
         "hotel india juliet kilo lima mike november oscar",
         "lima mike november oscar papa quebec"]
FULL = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec"


def tone(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        # Speech-like: 0.6 s bursts and 0.4 s pauses, so the loudness check finds no problem.
        wav.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(i / 8)) if (i / rate) % 1 < 0.6 else 0)
                                 for i in range(int(seconds * rate))))
    return path


def reply(kind, text, status=200):
    if kind == "gemini":
        return status, {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}],
                        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2}}
    words = text.split()
    return status, {"text": text, "segments": [{"start": 0.5, "end": 1.0 + len(words) * 0.2, "text": text}]}


@contextmanager
def gateway(kind, texts=None, statuses=()):
    """Answers the n-th transcription request with texts[n]; merges echo both windows."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            transcription = b"inline_data" in body or b"multipart" in self.headers.get("Content-Type", "").encode()
            seen.append({"path": self.path, "body": body, "transcription": transcription})
            count = sum(1 for s in seen if s["transcription"])
            status = statuses[len(seen) - 1] if len(seen) <= len(statuses) else 200
            if transcription:
                status, data = reply(kind, (texts or ["hello world"])[min(count, len(texts or [1])) - 1], status)
            else:
                status, data = reply("gemini", "merged boundary text", status)
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
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def spec_for(kind, url, **extra):
    model = {"gemini": "gemini-3.5-flash-lite", "groq": "whisper-large-v3", "http": "parakeet"}[kind]
    return ModelSpec(name=f"{kind}-model", kind=kind, model=model,
                     endpoint=f"{url}/v1/audio/transcriptions" if kind == "http" else None, **extra)


@pytest.fixture
def point_clients_at(monkeypatch):
    def apply(url):
        monkeypatch.setenv("GEMINI_BASE_URL", url)
        monkeypatch.setattr(core.GroqAPI, "API_URL", f"{url}/openai/v1/audio/transcriptions")
    return apply


def test_model_spec_from_config_options():
    spec = ModelSpec.from_options("local", {"kind": "http", "model": "parakeet", "endpoint": "http://h/v1",
                                            "prompt": "p", "temperature": "0.2", "chunk_seconds": "600"})
    assert (spec.kind, spec.model, spec.endpoint, spec.prompt, spec.temperature, spec.chunk_seconds) == (
        "http", "parakeet", "http://h/v1", "p", 0.2, 600.0)
    assert ModelSpec.from_options("gemini-3.6-flash", {"kind": "gemini"}).model == "gemini-3.6-flash"


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        ModelSpec.from_options("x", {"kind": "carrier-pigeon"})


@pytest.mark.parametrize("kind", ["gemini", "groq", "http"])
def test_short_file_has_the_common_result_shape(tmp_path, point_clients_at, kind):
    audio = tone(tmp_path / "a.wav", 2)
    with gateway(kind) as (url, seen):
        point_clients_at(url)
        result = engine.transcribe_file(audio, spec_for(kind, url), language="en", limiter=core.RateLimiter({}),
                                        timestamps="none")
    assert result["text"] == "hello world" and result["success"] is True and result["no_speech"] is False
    assert result["provider"] == kind and result["chunks"] == 1
    assert result["settings"]["language"] == "en"
    assert result["audio"]["source"] == str(audio.resolve())
    assert len([s for s in seen if s["transcription"]]) == 1


def test_a_gemini_prompt_replaces_the_built_in_instructions(tmp_path, point_clients_at):
    audio = tone(tmp_path / "a.wav", 2)
    with gateway("gemini") as (url, seen):
        point_clients_at(url)
        engine.transcribe_file(audio, spec_for("gemini", url, prompt="Write a verbatim transcript."),
                               language="ru", speakers=False, sounds=False, timestamps="none",
                               limiter=core.RateLimiter({}))
    body = json.loads(next(s for s in seen if s["transcription"])["body"])
    text = next(p["text"] for p in body["contents"][0]["parts"] if "text" in p)
    assert text.startswith("Write a verbatim transcript. ")
    assert "Transcribe the following audio" not in text and "ru language" not in text


@needs_ffmpeg
@pytest.mark.parametrize("kind", ["gemini", "groq", "http"])
def test_long_recording_is_chunked_and_reconstructed_for_every_kind(tmp_path, point_clients_at, kind):
    audio = tone(tmp_path / "long.wav", 25)
    with gateway(kind, PARTS) as (url, seen):
        point_clients_at(url)
        result = engine.transcribe_file(audio, spec_for(kind, url, chunk_seconds=10), overlap_seconds=2,
                                        limiter=core.RateLimiter({}), timestamps="none")
    assert len([s for s in seen if s["transcription"]]) == 4
    assert result["chunks"] == 4
    assert result["text"] == FULL
    assert [b["method"] for b in result["boundaries"]] == ["overlap-match"] * 3


@needs_ffmpeg
def test_chunk_segments_are_offset_to_recording_time(tmp_path, point_clients_at):
    audio = tone(tmp_path / "long.wav", 25)
    with gateway("groq", PARTS) as (url, _):
        point_clients_at(url)
        result = engine.transcribe_file(audio, spec_for("groq", url, chunk_seconds=10), overlap_seconds=2)
    starts = [s["start"] for s in result["segments"]]
    assert starts == sorted(starts) and starts[0] == pytest.approx(0.5) and starts[-1] > 18


@pytest.mark.parametrize("kind", ["gemini", "groq", "http"])
def test_transient_errors_are_retried_and_counted_by_the_shared_limiter(tmp_path, point_clients_at, kind):
    audio = tone(tmp_path / "a.wav", 2)
    spec = None
    with gateway(kind, statuses=(503, 200)) as (url, seen):
        point_clients_at(url)
        spec = spec_for(kind, url)
        limiter = core.RateLimiter({spec.model: core.Limits(rpd=50)})
        result = engine.transcribe_file(audio, spec, limiter=limiter, retry_base_delay=0, timestamps="none")
    assert result["text"] == "hello world"
    assert len(seen) == 2
    assert limiter.usage(spec.model)["today"] == 2


@needs_ffmpeg
def test_http_input_in_an_unsupported_container_is_converted(tmp_path, point_clients_at):
    import subprocess
    source = tone(tmp_path / "a.wav", 2)
    audio = tmp_path / "a.ogg"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), str(audio)], check=True)
    with gateway("http") as (url, seen):
        result = engine.transcribe_file(audio, spec_for("http", url))
    assert b'filename="audio.wav"' in seen[0]["body"]
    assert result["audio"]["converted"] is True


def test_unknown_duration_sends_one_request_and_warns(tmp_path, point_clients_at):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"not really audio")
    with gateway("http") as (url, seen):
        result = engine.transcribe_file(audio, spec_for("http", url))
    assert len(seen) == 1 and any("duration" in w for w in result["warnings"])


def test_http_failure_raises_a_typed_error(tmp_path):
    audio = tone(tmp_path / "a.wav", 2)
    with gateway("http", statuses=(403,)) as (url, _):
        with pytest.raises(core.ProviderError, match="403"):
            engine.transcribe_file(audio, spec_for("http", url))


@needs_ffmpeg
@pytest.mark.parametrize("kind", ["gemini", "groq", "http"])
def test_cli_and_library_give_the_same_transcript(tmp_path, point_clients_at, capsys, kind):
    audio = tone(tmp_path / "long.wav", 25)
    with gateway(kind, PARTS) as (url, _):
        point_clients_at(url)
        library = engine.transcribe_file(audio, spec_for(kind, url, chunk_seconds=10), overlap_seconds=2,
                                         limiter=core.RateLimiter({}))
    spec = spec_for(kind, "http://unused")
    config = tmp_path / "t.ini"
    with gateway(kind, PARTS) as (url, seen):
        point_clients_at(url)
        config.write_text(f"[defaults]\noverlap_seconds = 2\n[models]\nranking = chosen\n"
                          f"[model.chosen]\nkind = {kind}\nmodel = {spec.model}\nchunk_seconds = 10\n"
                          f"endpoint = {url}/v1/audio/transcriptions\n")
        assert runner.main([str(audio), "--config", str(config), "-m", "chosen"]) == 0
    assert capsys.readouterr().out == library["text"] + "\n"
    assert len([s for s in seen if s["transcription"]]) == 4


def test_transcribe_large_uses_the_shared_path(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(engine, "transcribe_file", lambda audio, spec, **kw: calls.append((spec, kw)) or {"text": "t"})
    monkeypatch.setattr(engine, "get_duration", lambda path: 60.0)
    audio = tmp_path / "a.m4a"
    audio.touch()
    api = object()
    core.transcribe_large(audio, api=api, model="gemini-3.5-flash-lite")
    spec, kw = calls[0]
    assert (spec.kind, spec.model) == ("gemini", "gemini-3.5-flash-lite") and kw["gemini_api"] is api


# ------------------------------------------------------------------- chunk plan

@pytest.mark.parametrize("duration", [2500.0, 3600.0, 3611.0, 5400.0, 7300.0])
def test_no_chunk_exceeds_its_limit_and_chunks_cover_the_recording(tmp_path, monkeypatch, duration):
    monkeypatch.setattr(audio_chunker, "_run_ffmpeg", lambda args: None)
    monkeypatch.setattr(audio_chunker, "get_duration", lambda path: duration)
    source = tmp_path / "a.m4a"
    source.touch()
    chunks = audio_chunker.split_audio(str(source), chunk_sec=1800, overlap_sec=10)
    assert chunks[0].start_time == 0 and chunks[-1].end_time == duration
    for chunk in chunks:
        assert chunk.end_time - chunk.start_time <= 1810 + 1e-6
        assert chunk.end_time - chunk.start_time > 20
    for left, right in zip(chunks, chunks[1:]):
        assert left.end_time - right.start_time == pytest.approx(10)


def test_long_live_recordings_are_split_into_short_sessions(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    audio = tmp_path / "diary.wav"
    audio.touch()
    seen = {}

    def split(path, chunk_sec, overlap_sec, output_dir):
        seen.update(chunk_sec=chunk_sec, overlap_sec=overlap_sec)
        return [MagicMock(path=str(audio), start_time=i * 500.0, end_time=i * 500.0 + 510) for i in range(3)]

    texts = iter(["one two three four five six", "three four five six seven eight", "five six seven eight nine ten"])
    monkeypatch.setattr(engine, "get_duration", lambda path: 1500.0)
    monkeypatch.setattr(engine, "split_audio", split)
    monkeypatch.setattr(engine, "_request", lambda *a: {"text": next(texts), "no_speech": False})
    spec = ModelSpec(name="live", kind="gemini", model="gemini-3.5-transcribe-live")
    result = engine.transcribe_file(audio, spec, prompt="ignored", language="ru")
    assert seen["chunk_sec"] + seen["overlap_sec"] <= engine.CHUNK_SECONDS["gemini-live"]
    assert result["text"] == "one two three four five six seven eight nine ten"
    assert any("Live models take no prompt" in w for w in result["warnings"])


def test_a_dropped_live_session_is_retried_once(tmp_path, monkeypatch):
    audio = tmp_path / "diary.wav"
    audio.touch()
    calls = []

    def request(spec, path, settings, clients):
        calls.append(path)
        if len(calls) == 1:
            raise core.GeminiAPIError("Live session is ending before transcription completed")
        return {"text": "hello there", "no_speech": False}
    monkeypatch.setattr(engine, "get_duration", lambda path: 60.0)
    monkeypatch.setattr(engine, "_request", request)
    spec = ModelSpec(name="live", kind="gemini", model="gemini-3.5-transcribe-live")
    result = engine.transcribe_file(audio, spec)
    assert result["text"] == "hello there" and len(calls) == 2
    assert any("retried" in w for w in result["warnings"])


def test_rest_parts_are_not_retried_by_the_engine(tmp_path, monkeypatch):
    """REST clients already retry 429/5xx; a permanent error must not be re-sent."""
    audio = tmp_path / "a.wav"
    audio.touch()
    calls = []

    def request(spec, path, settings, clients):
        calls.append(path)
        raise core.GeminiAPIError("API error 400")
    monkeypatch.setattr(engine, "get_duration", lambda path: 60.0)
    monkeypatch.setattr(engine, "_request", request)
    with pytest.raises(core.GeminiAPIError):
        engine.transcribe_file(audio, ModelSpec(name="g", kind="gemini", model="gemini-3.5-flash-lite"))
    assert len(calls) == 1
