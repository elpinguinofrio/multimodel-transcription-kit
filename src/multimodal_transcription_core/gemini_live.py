"""File transcription over Gemini's bidirectional Live WebSocket protocol.

The synchronous API matches ``transcribe``. Only this module's call path imports
the optional WebSocket dependency; importing the package has no side effects.
"""

import base64
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlencode, urlsplit, urlunsplit
import wave

from .gemini_client import GeminiAPI, GeminiAPIError, GeminiModel, assert_model_allowed
from .rate_limiter import estimate_audio_tokens
from .usage import track


LIVE_MODELS = {
    "gemini-3.8-live": "AUDIO",
    "gemini-3.5-transcribe-live": "TEXT",
}
_WS_PATH = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
_BYTES_PER_SECOND = 32000  # 16 kHz, mono, signed 16-bit little-endian
_CHUNK_BYTES = 3200  # 100 ms


@contextmanager
def _pcm_audio(path):
    """Prepare PCM on temporary local disk without overwriting source files."""
    with tempfile.TemporaryFile() as pcm:
        native = False
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as audio:
                    if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
                            audio.getcomptype()) == (1, 2, 16000, "NONE"):
                        native = True
                        while chunk := audio.readframes(16000):
                            pcm.write(chunk)
            except (wave.Error, EOFError):
                pass
        if not native:
            pcm.seek(0)
            pcm.truncate()
            try:
                converted = subprocess.run(
                    ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                     "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
                    stdout=pcm, stderr=subprocess.PIPE, check=False,
                )
            except FileNotFoundError as exc:
                raise GeminiAPIError("ffmpeg is required to prepare this audio for Live") from exc
            if converted.returncode:
                raise GeminiAPIError("Failed to convert audio to Live PCM")
        size = pcm.seek(0, 2)
        if not size or size % 2:
            raise GeminiAPIError("Live audio must contain complete PCM samples")
        pcm.seek(0)
        yield pcm, size / _BYTES_PER_SECOND


def _websocket_url(api):
    parts = urlsplit(api.base_url)
    scheme = {"https": "wss", "http": "ws", "wss": "wss", "ws": "ws"}.get(parts.scheme)
    if not scheme or not parts.netloc or parts.query or parts.fragment or parts.username:
        raise ValueError("GEMINI_BASE_URL must be an HTTP(S) or WS(S) base URL")
    return urlunsplit((scheme, parts.netloc, parts.path.rstrip("/") + _WS_PATH,
                       urlencode({"key": api.api_key}), ""))


def transcribe_live(audio_path, model, *, api=None):
    """Stream a recording in real time and return the standard result dictionary.

    Supported models: gemini-3.8-live and gemini-3.5-transcribe-live. Uses the
    configured GeminiAPI's key, base URL, timeout and shared limiter. Requires
    the ``live`` extra; non-native audio also requires ffmpeg. Failures raise
    GeminiAPIError (quota errors propagate). Late text is drained for one quiet
    second after turn completion; longer delays cannot be guaranteed. This is a
    single session, with no automatic replay or model fallback.
    """
    model_name = model.value if isinstance(model, GeminiModel) else str(model)
    assert_model_allowed(model_name)
    if model_name not in LIVE_MODELS:
        raise ValueError(f"Unsupported Live model: {model_name}")
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        from websockets.sync.client import connect
    except ImportError as exc:
        raise GeminiAPIError("Live transcription requires multimodal-transcription-core[live]") from exc
    api = api if api is not None else GeminiAPI()
    if api.timeout <= 0:
        raise ValueError("Live timeout must be positive")
    url = _websocket_url(api)

    with _pcm_audio(path) as (pcm, duration):
        if api.rate_limiter is not None:
            api.rate_limiter.acquire(model_name, estimate_audio_tokens(duration))
        else:
            api._throttle()
            api._rate_limit()
        started = time.monotonic()
        setup = {
            "model": f"models/{model_name}",
            "generationConfig": {"responseModalities": [LIVE_MODELS[model_name]]},
            "inputAudioTranscription": {},
            "realtimeInputConfig": {"automaticActivityDetection": {"disabled": True}},
        }
        try:
            with track(api.usage_log, "gemini", model_name, "live") as event:
                usage = {}
                event["response"] = {"usageMetadata": usage}
                with connect(url, open_timeout=api.timeout, close_timeout=1) as ws:
                    ws.send(json.dumps({"setup": setup}))
                    message = _receive(ws, api.timeout)
                    if "setupComplete" not in message:
                        raise GeminiAPIError("Live server did not acknowledge setup")
                    return _stream(ws, pcm, duration, model_name, api.timeout, started, usage=usage)
        except GeminiAPIError:
            raise
        except Exception as exc:
            # Exceptions can include authenticated URLs or echoed credentials,
            # including URL-encoded forms. Keep only the exception type.
            raise GeminiAPIError(f"Live transcription failed ({type(exc).__name__})") from None


def _receive(ws, timeout):
    message = json.loads(ws.recv(timeout=timeout))
    if not isinstance(message, dict):
        raise GeminiAPIError("Invalid Live server message")
    if "error" in message:
        # Don't expose untrusted server diagnostics (which can echo credentials).
        raise GeminiAPIError("Live server rejected the request")
    if "goAway" in message:
        raise GeminiAPIError("Live session is ending before transcription completed")
    return message


def _stream(ws, pcm, duration, model, timeout, started, usage=None):
    stop = threading.Event()
    end_started = threading.Event()
    ended = threading.Event()
    send_errors = []

    def send_audio():
        try:
            ws.send(json.dumps({"realtimeInput": {"activityStart": {}}}))
            stream_start = time.monotonic()
            sent_bytes = 0
            while chunk := pcm.read(_CHUNK_BYTES):
                if stop.wait(max(0, stream_start + sent_bytes / _BYTES_PER_SECOND - time.monotonic())):
                    return
                ws.send(json.dumps({"realtimeInput": {"audio": {
                    "mimeType": "audio/pcm;rate=16000",
                    "data": base64.b64encode(chunk).decode("ascii"),
                }}}))
                sent_bytes += len(chunk)
            if not stop.wait(max(0, stream_start + duration - time.monotonic())):
                end_started.set()
                ws.send(json.dumps({"realtimeInput": {"activityEnd": {}}}))
                ended.set()
        except Exception as exc:
            send_errors.append(exc)

    sender = threading.Thread(target=send_audio, name="gemini-live-audio", daemon=True)
    sender.start()
    texts = []
    usage = {} if usage is None else usage
    turn_complete = False
    transcript_pending = False
    deadline = time.monotonic() + duration + timeout
    settle_deadline = None
    try:
        while True:
            if send_errors:
                raise GeminiAPIError("Failed to send Live audio")
            now = time.monotonic()
            if now >= deadline:
                raise GeminiAPIError("Timed out waiting for complete Live transcription")
            if settle_deadline is not None and now >= settle_deadline and not transcript_pending:
                break
            try:
                message = _receive(ws, min(0.1, deadline - now))
            except TimeoutError:
                continue
            usage.update(message.get("usageMetadata", {}))
            content = message.get("serverContent", {})
            transcript = content.get("inputTranscription", {})
            if "finished" in transcript:
                transcript_pending = transcript["finished"] is False
            if isinstance(transcript.get("text"), str):
                texts.append(transcript["text"])
                if turn_complete:
                    settle_deadline = time.monotonic() + min(1.0, timeout)
            if end_started.is_set():
                turn_complete |= bool(content.get("turnComplete"))
            if turn_complete and settle_deadline is None:
                # Input transcripts are independently ordered relative to turns.
                # The protocol has no terminal input-transcript marker, so allow
                # one quiet second for late fragments after turnComplete.
                settle_deadline = time.monotonic() + min(1.0, timeout)
        sender.join(timeout=1)
        if send_errors:
            raise GeminiAPIError("Failed to send Live audio")
        if sender.is_alive() or not ended.is_set():
            raise GeminiAPIError("Live audio sender did not complete")
        if not texts:
            raise GeminiAPIError("Live session completed without an input transcript")
        return {
            "text": "".join(texts).strip(), "model": model, "duration": duration,
            "processing_time": time.monotonic() - started,
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("responseTokenCount", 0), "success": True,
        }
    finally:
        stop.set()
        ws.close()
        sender.join(timeout=2)
