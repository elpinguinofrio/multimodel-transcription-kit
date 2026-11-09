"""A failed part is retried by the next models, for that part only; the rest stay as they are.

Seen live 2026-11-09: flash-lite called 2 of 9 parts silent and the whole 7-minute file
then went to gemini-3.8-flash, which had 2 other parts blocked. Replies are faked;
ffmpeg is real (splitting and the speech check).
"""
import shutil

import pytest

from multimodel_transcription_kit import engine, selection
from multimodel_transcription_kit.engine import ModelSpec
from multimodel_transcription_kit.errors import IncompleteTranscriptError, NoModelAvailableError, ProviderError
from test_engine import FULL, PARTS, tone

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                reason="requires ffmpeg and ffprobe")
SILENCE = {"text": "[SILENCE]", "no_speech": True}
OPTIONS = dict(overlap_seconds=2, timestamps="none", speakers=False, sounds=False, enhance=False)


def spec(name):
    return ModelSpec(name=name, kind="gemini", model=f"model-{name}", chunk_seconds=10)


def fake(monkeypatch, plan):
    """plan[name] = function(part index from 0, call count) -> reply or exception."""
    calls = {}

    def request(spec, path, settings, clients):
        index = part_index(path)
        calls.setdefault(spec.name, []).append(index)
        reply = plan[spec.name](index)
        if isinstance(reply, Exception):
            raise reply
        return dict(reply)
    monkeypatch.setattr(engine, "_request", request)
    return calls


_ORDER = {}   # chunk path -> part index (0-based), filled by the recording fixture


def part_index(path):
    return _ORDER[str(path)]


@pytest.fixture
def recording(tmp_path, monkeypatch):
    audio = tone(tmp_path / "long.wav", 25)
    real_split = engine.split_audio

    def split(*args, **kwargs):
        chunks = real_split(*args, **kwargs)
        _ORDER.clear()
        _ORDER.update({c.path: i for i, c in enumerate(chunks)})
        return chunks
    monkeypatch.setattr(engine, "split_audio", split)
    return audio


def speech(index):
    return {"text": PARTS[index], "no_speech": False}


def test_a_failed_part_is_retried_by_the_next_model_and_the_rest_are_kept(recording, monkeypatch):
    calls = fake(monkeypatch, {"lite": lambda i: SILENCE if i == 1 else speech(i), "flash": speech})
    result = engine.transcribe_file(recording, spec("lite"), part_fallbacks=[spec("flash")], **OPTIONS)
    assert result["text"] == FULL
    assert calls == {"lite": [0, 1, 2, 3], "flash": [1]}
    assert any("part 2" in w and "flash" in w for w in result["warnings"])


def test_fallbacks_are_tried_in_order(recording, monkeypatch):
    calls = fake(monkeypatch, {"lite": lambda i: ProviderError("PROMPT BLOCKED: SAFETY") if i == 2 else speech(i),
                               "flash": lambda i: ProviderError("PROMPT BLOCKED: SAFETY"),
                               "pro": speech})
    result = engine.transcribe_file(recording, spec("lite"), part_fallbacks=[spec("flash"), spec("pro")], **OPTIONS)
    assert result["text"] == FULL and calls["flash"] == [2] and calls["pro"] == [2]


def test_a_part_no_model_can_transcribe_fails_the_recording(recording, monkeypatch):
    fake(monkeypatch, {"lite": lambda i: SILENCE if i == 1 else speech(i), "flash": lambda i: SILENCE})
    with pytest.raises(IncompleteTranscriptError) as caught:
        engine.transcribe_file(recording, spec("lite"), part_fallbacks=[spec("flash")], **OPTIONS)
    assert caught.value.failed_parts == [2]


def test_without_fallbacks_a_failed_part_fails_as_before(recording, monkeypatch):
    fake(monkeypatch, {"lite": lambda i: SILENCE if i == 1 else speech(i)})
    with pytest.raises(IncompleteTranscriptError):
        engine.transcribe_file(recording, spec("lite"), **OPTIONS)


def test_a_fallback_that_cannot_supply_the_requested_times_is_skipped(recording, monkeypatch):
    timed = lambda i: {"text": PARTS[i], "no_speech": False,
                       "segments": [{"start": 0.5, "end": 2.0, "text": PARTS[i]}]}
    calls = fake(monkeypatch, {"whisper": lambda i: SILENCE if i == 1 else timed(i), "flash": speech,
                               "parakeet": timed})
    whisper = ModelSpec(name="whisper", kind="groq", model="whisper", chunk_seconds=10)
    parakeet = ModelSpec(name="parakeet", kind="groq", model="parakeet", chunk_seconds=10)
    result = engine.transcribe_file(recording, whisper, part_fallbacks=[spec("flash"), parakeet],
                                    **{**OPTIONS, "timestamps": "phrase"})
    assert "flash" not in calls and calls["parakeet"] == [1]
    assert result["segments"]


def test_selection_retries_only_the_failed_part_with_the_next_model(recording, monkeypatch):
    calls = fake(monkeypatch, {"lite": lambda i: SILENCE if i == 1 else speech(i), "flash": speech})
    result = selection.transcribe_best(recording, [spec("lite"), spec("flash")], probe=lambda s: True, **OPTIONS)
    assert result["text"] == FULL and result["selection"]["chosen"] == "lite"
    assert calls["flash"] == [1]


def test_selection_does_not_redo_the_recording_after_every_model_failed_a_part(recording, monkeypatch):
    calls = fake(monkeypatch, {"lite": lambda i: SILENCE if i == 1 else speech(i), "flash": lambda i: SILENCE})
    with pytest.raises(NoModelAvailableError):
        selection.transcribe_best(recording, [spec("lite"), spec("flash")], probe=lambda s: True, **OPTIONS)
    assert calls["flash"] == [1]


def test_selection_offers_only_reachable_models_as_part_fallbacks(recording, monkeypatch):
    calls = fake(monkeypatch, {"lite": lambda i: SILENCE if i == 1 else speech(i), "down": speech, "flash": speech})
    result = selection.transcribe_best(recording, [spec("lite"), spec("down"), spec("flash")],
                                       probe=lambda s: s.name != "down", **OPTIONS)
    assert result["text"] == FULL and "down" not in calls and calls["flash"] == [1]


def test_the_error_says_why_each_model_failed_the_part(recording, monkeypatch):
    fake(monkeypatch, {"lite": lambda i: SILENCE if i == 1 else speech(i),
                       "flash": lambda i: ProviderError("PROMPT BLOCKED: SAFETY")})
    with pytest.raises(IncompleteTranscriptError) as caught:
        engine.transcribe_file(recording, spec("lite"), part_fallbacks=[spec("flash")], **OPTIONS)
    message = str(caught.value)
    assert "part 2" in message and "SILENCE" in message and "flash" in message and "SAFETY" in message


def test_selection_reports_why_the_part_failed(recording, monkeypatch):
    fake(monkeypatch, {"lite": lambda i: SILENCE if i == 1 else speech(i),
                       "flash": lambda i: ProviderError("PROMPT BLOCKED: SAFETY")})
    with pytest.raises(NoModelAvailableError, match="SAFETY"):
        selection.transcribe_best(recording, [spec("lite"), spec("flash")], probe=lambda s: True, **OPTIONS)
