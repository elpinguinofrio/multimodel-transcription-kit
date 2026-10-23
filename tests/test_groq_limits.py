"""Groq waits on 429 instead of failing, and resends the whole file."""
from unittest.mock import MagicMock, patch

import requests

from multimodel_transcription_kit import GroqAPI, Limits, RateLimiter


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
         patch("multimodel_transcription_kit.audio_converter.AudioConverter.get_audio_duration",
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
         patch("multimodel_transcription_kit.audio_converter.AudioConverter.get_audio_duration",
               return_value=1.0):
        GroqAPI(rate_limiter=rl).speech_to_text(str(audio))
    assert rl.usage("whisper-large-v3")["today"] == 1
