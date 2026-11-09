"""A "[SILENCE]" reply for audio that clearly holds speech is a failed request, not a transcript.

Seen live 2026-11-09: gemini-3.5-flash-lite answered [SILENCE] for the first minute of
the 120 s diary sample (speech -23 dB, SNR 22 dB). Replies are faked; ffmpeg is real.
"""
import shutil
import subprocess

import pytest

from multimodel_transcription_kit import engine, selection
from multimodel_transcription_kit.engine import ModelSpec
from multimodel_transcription_kit.errors import IncompleteTranscriptError, ProviderError
from test_engine import PARTS, tone

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="requires ffmpeg and ffprobe")
SILENCE = {"text": "[SILENCE]", "no_speech": True}


def lavfi(path, source, seconds=5):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", source,
                    "-t", str(seconds), "-ac", "1", "-ar", "16000", str(path)], check=True)
    return path


def spec(name="g", **extra):
    return ModelSpec(name=name, kind="gemini", model="gemini-3.5-flash-lite", **extra)


def run(audio, monkeypatch, reply, **extra):
    monkeypatch.setattr(engine, "_request", lambda spec, path, settings, clients: dict(reply))
    return engine.transcribe_file(audio, spec(**extra), timestamps="none", speakers=False, sounds=False,
                                  enhance=False)


@needs_ffmpeg
def test_silence_reply_for_audio_with_speech_fails(tmp_path, monkeypatch):
    with pytest.raises(ProviderError, match=r"\[SILENCE\].*speech"):
        run(tone(tmp_path / "speech.wav", 4), monkeypatch, SILENCE)


@needs_ffmpeg
@pytest.mark.parametrize("source", ["anullsrc=r=16000:cl=mono",                 # digital silence
                                    "anoisesrc=r=16000:a=0.003:c=pink",          # quiet room
                                    "anoisesrc=r=16000:a=0.03:c=pink"])          # steady hiss
def test_silence_reply_for_silent_or_noise_only_audio_stands(tmp_path, monkeypatch, source):
    result = run(lavfi(tmp_path / "quiet.wav", source), monkeypatch, SILENCE)
    assert result["no_speech"] is True and result["success"] is True


@needs_ffmpeg
def test_speech_reply_is_not_checked(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(engine, "assess_audio", lambda path: calls.append(path) or {"problems": []})
    result = run(tone(tmp_path / "speech.wav", 4), monkeypatch, {"text": "hello there", "no_speech": False})
    assert result["text"] == "hello there" and len(calls) == 1      # only the up-front loudness check


@needs_ffmpeg
def test_a_falsely_silent_part_of_a_long_recording_is_a_failed_part(tmp_path, monkeypatch):
    audio = tone(tmp_path / "long.wav", 25)
    replies = iter([SILENCE] + [{"text": text, "no_speech": False} for text in PARTS[1:]])
    monkeypatch.setattr(engine, "_request", lambda *a: dict(next(replies)))
    with pytest.raises(IncompleteTranscriptError) as caught:
        engine.transcribe_file(audio, spec(chunk_seconds=10), overlap_seconds=2, timestamps="none",
                               speakers=False, sounds=False, enhance=False)
    assert caught.value.failed_parts == [1]


@needs_ffmpeg
def test_the_next_model_transcribes_when_the_first_says_silence_over_speech(tmp_path, monkeypatch):
    audio = tone(tmp_path / "speech.wav", 4)
    monkeypatch.setattr(engine, "_request", lambda spec, *a: (
        dict(SILENCE) if spec.name == "lite" else {"text": "the words", "no_speech": False}))
    result = selection.transcribe_best(audio, [spec("lite"), spec("flash")], probe=lambda s: True,
                                       timestamps="none", speakers=False, sounds=False, enhance=False)
    assert result["text"] == "the words"
    assert result["selection"]["chosen"] == "flash"
    assert "SILENCE" in result["selection"]["attempts"][0]["error"]
