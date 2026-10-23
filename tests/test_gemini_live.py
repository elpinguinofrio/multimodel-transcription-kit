"""Exercise the production Live transport against a synthetic WebSocket gateway."""

import base64
from contextlib import contextmanager
import json
import threading
import time
import wave

import pytest

import multimodel_transcription_kit as core


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "synthetic.wav"
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"\x01\x00" * 3680)  # 230 ms
    return path


@contextmanager
def gateway(mode="success", modality="TEXT"):
    server_module = pytest.importorskip("websockets.sync.server")
    from websockets.exceptions import ConnectionClosed

    observed = {"audio": bytearray(), "times": [], "errors": []}

    def handler(ws):
        try:
            setup = json.loads(ws.recv(timeout=2))["setup"]
            observed["setup"] = setup
            assert setup["generationConfig"]["responseModalities"] == [modality]
            assert setup["realtimeInputConfig"]["automaticActivityDetection"]["disabled"]
            if mode == "setup_error":
                ws.send(json.dumps({"error": {"code": 400, "message": "bad setup"}}))
                return
            ws.send(json.dumps({"setupComplete": {}}))
            assert json.loads(ws.recv(timeout=2)) == {"realtimeInput": {"activityStart": {}}}
            while True:
                message = json.loads(ws.recv(timeout=2))["realtimeInput"]
                if "activityEnd" in message:
                    observed["end"] = time.monotonic()
                    break
                chunk = message["audio"]
                assert chunk["mimeType"] == "audio/pcm;rate=16000"
                data = base64.b64decode(chunk["data"])
                assert 0 < len(data) <= 3200
                observed["audio"].extend(data)
                observed["times"].append(time.monotonic())
                if len(observed["times"]) == 1:
                    if mode == "early_generation":
                        ws.send(json.dumps({"serverContent": {"generationComplete": True}}))
                    if mode == "stream_error":
                        ws.send(json.dumps({"error": {"message": "stream rejected"}}))
                        return
                    if mode == "disconnect":
                        ws.close(1011, "upstream failure")
                        return
                    if mode == "go_away":
                        ws.send(json.dumps({"goAway": {"timeLeft": "0s"}}))
                        return
                    if mode != "timeout":
                        ws.send(json.dumps({"serverContent": {
                            "interimInputTranscription": {"text": "wrong interim"},
                            "inputTranscription": {"text": "Hello "},
                            "outputTranscription": {"text": "assistant reply"},
                        }}))
            if mode == "timeout":
                try:
                    ws.recv(timeout=2)
                except ConnectionClosed:
                    pass
                return
            completion = "generationComplete" if mode in ("generation_only", "generation_unfinished") else "turnComplete"
            if mode != "early_generation":
                ws.send(json.dumps({"serverContent": {completion: True}}))
            # Input transcripts are independently ordered: the last can arrive
            # after turnComplete. An early-return implementation loses this.
            time.sleep(0.02)
            ws.send(json.dumps({
                "serverContent": {"inputTranscription": {
                    "text": "world", **({"finished": False} if mode in ("unfinished", "generation_unfinished") else
                                         {"finished": True} if mode == "finished" else {}),
                }},
                "usageMetadata": {"promptTokenCount": 7, "responseTokenCount": 3},
            }))
            try:
                ws.recv(timeout=2)
            except ConnectionClosed:
                pass
        except ConnectionClosed:
            pass
        except Exception as exc:
            observed["errors"].append(exc)

    with server_module.serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.socket.getsockname()[1]}", observed
        finally:
            server.shutdown()
            thread.join(timeout=3)
    assert not observed["errors"]


def live_function():
    function = getattr(core, "transcribe_live", None)
    assert callable(function), "The shared core must export transcribe_live"
    return function


def test_transcribe_live_accepts_generation_completion_after_audio_ends(audio, monkeypatch):
    # Real Transcribe Live sessions can finish with generationComplete and no turnComplete.
    with gateway(mode="generation_only") as (base_url, _):
        monkeypatch.setenv("GEMINI_BASE_URL", base_url)
        result = live_function()(audio, "gemini-3.5-transcribe-live",
                                 api=core.GeminiAPI(throttle_rate=0, timeout=2))
    assert result["text"] == "Hello world"


def test_generation_completion_cannot_accept_an_unfinished_transcript(audio, monkeypatch):
    with gateway(mode="generation_unfinished") as (base_url, _):
        monkeypatch.setenv("GEMINI_BASE_URL", base_url)
        with pytest.raises(core.GeminiAPIError, match="Timed out"):
            live_function()(audio, "gemini-3.5-transcribe-live",
                            api=core.GeminiAPI(throttle_rate=0, timeout=2))


def test_audio_output_live_still_requires_turn_completion(audio, monkeypatch):
    with gateway(mode="generation_only", modality="AUDIO") as (base_url, _):
        monkeypatch.setenv("GEMINI_BASE_URL", base_url)
        with pytest.raises(core.GeminiAPIError, match="Timed out"):
            live_function()(audio, "gemini-3.8-live", api=core.GeminiAPI(throttle_rate=0, timeout=2))


def test_generation_completion_before_audio_ends_is_not_terminal(audio, monkeypatch):
    with gateway(mode="early_generation") as (base_url, _):
        monkeypatch.setenv("GEMINI_BASE_URL", base_url)
        with pytest.raises(core.GeminiAPIError, match="Timed out"):
            live_function()(audio, "gemini-3.5-transcribe-live", api=core.GeminiAPI(throttle_rate=0, timeout=2))


@pytest.mark.parametrize("model,modality", [
    ("gemini-3.5-transcribe-live", "TEXT"), ("gemini-3.8-live", "AUDIO"),
])
def test_live_streams_pcm_at_real_time_and_collects_input_text(audio, monkeypatch, model, modality):
    transcribe_live = live_function()
    with gateway(modality=modality) as (base_url, observed):
        monkeypatch.setenv("GEMINI_BASE_URL", base_url)
        result = transcribe_live(audio, model, api=core.GeminiAPI(throttle_rate=0, timeout=2))
    assert result["text"] == "Hello world"
    assert result["success"] is True
    assert result["model"] == model
    assert result["duration"] == pytest.approx(0.23)
    assert result["input_tokens"] == 7
    assert result["output_tokens"] == 3
    assert bytes(observed["audio"]) == b"\x01\x00" * 3680
    assert len(observed["times"]) == 3
    assert observed["times"][1] - observed["times"][0] >= 0.09
    assert observed["times"][2] - observed["times"][0] >= 0.19
    assert observed["end"] - observed["times"][0] >= 0.22


@pytest.mark.parametrize("mode", ["setup_error", "stream_error", "disconnect", "go_away", "timeout"])
def test_live_never_returns_partial_success_on_failure(audio, monkeypatch, mode):
    transcribe_live = live_function()
    with gateway(mode=mode) as (base_url, observed):
        monkeypatch.setenv("GEMINI_BASE_URL", base_url)
        with pytest.raises(core.GeminiAPIError):
            transcribe_live(audio, "gemini-3.5-transcribe-live",
                            api=core.GeminiAPI(throttle_rate=0, timeout=0.3))


def test_existing_transcribe_routes_live_model(audio, monkeypatch):
    with gateway() as (base_url, observed):
        monkeypatch.setenv("GEMINI_BASE_URL", base_url)
        result = core.transcribe(audio, model="gemini-3.5-transcribe-live",
                                 api=core.GeminiAPI(throttle_rate=0, timeout=2))
    assert result["text"] == "Hello world"


def test_live_refuses_batch_model_before_network(audio):
    with pytest.raises(ValueError, match="Live model"):
        live_function()(audio, "gemini-3.5-flash-lite")


def test_live_preserves_quota_errors(audio, tmp_path):
    limiter = core.RateLimiter({"gemini-3.5-transcribe-live": core.Limits(rpd=0)},
                               state_path=tmp_path / "quota.sqlite")
    with pytest.raises(core.QuotaExhausted):
        live_function()(audio, "gemini-3.5-transcribe-live", api=core.GeminiAPI(rate_limiter=limiter))


def test_live_rejects_missing_audio(tmp_path):
    with pytest.raises(FileNotFoundError):
        live_function()(tmp_path / "missing.wav", "gemini-3.5-transcribe-live")


def test_live_options_do_not_silently_ignore_analysis(audio):
    with pytest.raises(RuntimeError, match="Live.*plain transcription"):
        core.transcribe(audio, model="gemini-3.5-transcribe-live", analyze=True)


def test_live_converts_stereo_8khz_wav(audio, monkeypatch):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    with wave.open(str(audio), "wb") as output:
        output.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        output.writeframes(b"\x01\x00\x01\x00" * 1840)
    with gateway() as (base_url, observed):
        monkeypatch.setenv("GEMINI_BASE_URL", base_url)
        result = live_function()(audio, "gemini-3.5-transcribe-live",
                                 api=core.GeminiAPI(throttle_rate=0, timeout=2))
    assert result["duration"] == pytest.approx(0.23)
    assert len(observed["audio"]) == 7360
    assert result["text"] == "Hello world"


def test_live_rejects_empty_audio(tmp_path):
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    with pytest.raises(core.GeminiAPIError, match="PCM"):
        live_function()(path, "gemini-3.5-transcribe-live")


def test_explicitly_unfinished_transcription_cannot_return_success(audio, monkeypatch):
    with gateway(mode="unfinished") as (base_url, observed):
        monkeypatch.setenv("GEMINI_BASE_URL", base_url)
        with pytest.raises(core.GeminiAPIError, match="Timed out"):
            live_function()(audio, "gemini-3.5-transcribe-live",
                            api=core.GeminiAPI(throttle_rate=0, timeout=1.5))


def test_chunked_live_is_rejected_before_duration_probe(audio, monkeypatch):
    from multimodel_transcription_kit import chunked_transcriber
    def forbidden(*args, **kwargs):
        pytest.fail("Live models must not enter the REST overlap-merge pipeline")
    monkeypatch.setattr(chunked_transcriber, "get_duration", forbidden)
    with pytest.raises(ValueError, match="transcribe_live"):
        core.transcribe_large(audio, model="gemini-3.5-transcribe-live")


def test_end_marker_send_failure_cannot_return_success(audio, monkeypatch):
    from websockets.sync import client
    connect = client.connect

    class FailingEnd:
        def __init__(self, ws):
            self.ws = ws

        def __getattr__(self, name):
            return getattr(self.ws, name)

        def send(self, message):
            self.ws.send(message)
            if "activityEnd" in json.loads(message).get("realtimeInput", {}):
                time.sleep(0.05)  # receiver can observe completion before send returns
                raise OSError("end-marker send failed")

    @contextmanager
    def failing_connect(*args, **kwargs):
        with connect(*args, **kwargs) as ws:
            yield FailingEnd(ws)

    monkeypatch.setattr(client, "connect", failing_connect)
    with gateway(mode="finished") as (base_url, observed):
        monkeypatch.setenv("GEMINI_BASE_URL", base_url)
        with pytest.raises(core.GeminiAPIError, match="send Live audio"):
            live_function()(audio, "gemini-3.5-transcribe-live",
                            api=core.GeminiAPI(throttle_rate=0, timeout=2))


def test_failed_handshake_does_not_expose_encoded_credentials(audio, monkeypatch):
    from websockets.sync import client
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic/token+reserved=")
    def fail(url, **kwargs):
        raise OSError(f"Handshake failed at {url}")
    monkeypatch.setattr(client, "connect", fail)
    with pytest.raises(core.GeminiAPIError) as error:
        live_function()(audio, "gemini-3.5-transcribe-live",
                        api=core.GeminiAPI(throttle_rate=0))
    assert "synthetic" not in str(error.value)
    assert "key=" not in str(error.value)
