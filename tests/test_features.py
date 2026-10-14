"""Speakers, non-speech sounds and timestamps: library arguments, CLI flags, defaults.

Decision (owner, 2026-10-14): a default-on feature the model cannot supply is
dropped with a warning recorded in the result; an explicitly requested one raises
UnsupportedFeatureError. Timestamps only come from provider alignment.
"""
import inspect
from unittest.mock import Mock

import pytest

import multimodal_transcription_core as core
from multimodal_transcription_core import engine, runner
from multimodal_transcription_core.engine import ModelSpec, resolve_features
from multimodal_transcription_core.render import render_timestamps

GEMINI = ModelSpec(name="g", kind="gemini", model="gemini-3.5-flash-lite")
GROQ = ModelSpec(name="w", kind="groq", model="whisper-large-v3")
HTTP = ModelSpec(name="p", kind="http", model="parakeet", endpoint="http://127.0.0.1:9/v1")
LIVE = ModelSpec(name="l", kind="gemini", model="gemini-3.5-transcribe-live")


def test_defaults_are_speakers_sounds_and_major_timestamps():
    applied, warnings = resolve_features(GEMINI)
    assert applied == {"speakers": True, "sounds": True, "timestamps": "none"}
    assert any("timestamps" in w and "alignment" in w for w in warnings)
    applied, warnings = resolve_features(GROQ)
    assert applied == {"speakers": False, "sounds": False, "timestamps": "major"}
    assert any("speaker" in w for w in warnings) and any("sound" in w for w in warnings)


@pytest.mark.parametrize("spec,request_", [
    (GROQ, {"speakers": True}), (HTTP, {"sounds": True}), (GEMINI, {"timestamps": "phrase"}),
    (GEMINI, {"timestamps": "major"}), (LIVE, {"speakers": True}),
])
def test_explicitly_requested_unsupported_feature_raises(spec, request_):
    with pytest.raises(core.UnsupportedFeatureError):
        resolve_features(spec, **request_)


def test_explicitly_disabled_features_need_no_capability():
    applied, warnings = resolve_features(GROQ, speakers=False, sounds=False, timestamps="none")
    assert applied == {"speakers": False, "sounds": False, "timestamps": "none"} and warnings == []


def test_configured_capabilities_override_the_kind_defaults():
    spec = ModelSpec.from_options("diarizer", {"kind": "http", "endpoint": "http://h/v1",
                                               "capabilities": "speakers, timestamps"})
    applied, warnings = resolve_features(spec, speakers=True)
    assert applied["speakers"] is True and applied["timestamps"] == "major"


def test_invalid_timestamp_mode_is_rejected():
    with pytest.raises(ValueError, match="timestamps"):
        resolve_features(GROQ, timestamps="every-word")


# ------------------------------------------------------------------ prompts

def prompt_sent(monkeypatch, **kwargs):
    api = core.GeminiAPI(throttle_rate=0)
    monkeypatch.setattr(api, "_rate_limit", lambda: None)
    sent = {}
    response = Mock(status_code=200)
    response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}

    def post(url, payload, **kw):
        sent["prompt"] = payload["contents"][0]["parts"][0]["text"]
        return response
    monkeypatch.setattr(api, "_post_with_retry", post)
    api.speech_to_text(b"synthetic", **kwargs)
    return sent["prompt"]


def test_sounds_on_asks_for_asterisk_sound_descriptions(monkeypatch):
    prompt = prompt_sent(monkeypatch, sounds=True)
    assert "*dog barking*" in prompt and "Output only the spoken words" not in prompt


def test_sounds_off_asks_for_spoken_words_only(monkeypatch):
    assert "do not describe non-speech sounds" in prompt_sent(monkeypatch, sounds=False).lower()


def test_speakers_with_sounds(monkeypatch):
    prompt = prompt_sent(monkeypatch, diarize=True, sounds=True)
    assert "'Speaker 1:'" in prompt and "*dog barking*" in prompt


def test_legacy_prompt_is_unchanged_without_the_new_argument(monkeypatch):
    prompt = prompt_sent(monkeypatch)
    assert prompt.startswith("Transcribe the following audio to plain text. "
                             "Do not include timestamps or time markers. Output only the spoken words.")


# ------------------------------------------------------------------ rendering

SEGMENTS = [{"start": 0.0, "end": 2.0, "text": "Hello."}, {"start": 2.2, "end": 4.0, "text": "How are you?"},
            {"start": 7.5, "end": 9.0, "text": "Fine."}, {"start": 3725.0, "end": 3726.0, "text": "Late."}]


def test_phrase_timestamps_mark_every_aligned_segment():
    assert render_timestamps(SEGMENTS, "phrase") == (
        "[0:00:00] Hello.\n[0:00:02] How are you?\n[0:00:07] Fine.\n[1:02:05] Late.")


def test_major_timestamps_mark_paragraphs_at_pauses():
    assert render_timestamps(SEGMENTS, "major") == (
        "[0:00:00] Hello. How are you?\n\n[0:00:07] Fine.\n\n[1:02:05] Late.")


def test_major_timestamps_start_a_paragraph_at_least_every_minute():
    segments = [{"start": float(i * 10), "end": float(i * 10 + 9.5), "text": f"s{i}"} for i in range(13)]
    rendered = render_timestamps(segments, "major")
    assert rendered.count("[") == 3 and "[0:01:00] s6" in rendered


# ------------------------------------------------------------------ engine + CLI

def fake_request(text="hello there", segments=None):
    return lambda spec, path, settings, clients: {"text": text, "segments": segments, "no_speech": False}


def test_engine_renders_alignment_timestamps_by_default(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.touch()
    monkeypatch.setattr(engine, "get_duration", lambda path: 10.0)
    monkeypatch.setattr(engine, "_request", fake_request("Hello. How are you?", SEGMENTS[:2]))
    result = engine.transcribe_file(audio, GROQ)
    assert result["text"] == "[0:00:00] Hello. How are you?"
    assert result["features"] == {"speakers": False, "sounds": False, "timestamps": "major"}
    assert engine.transcribe_file(audio, GROQ, timestamps="none")["text"] == "Hello. How are you?"


def test_missing_alignment_raises_when_timestamps_were_requested(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.touch()
    monkeypatch.setattr(engine, "get_duration", lambda path: 10.0)
    monkeypatch.setattr(engine, "_request", fake_request())
    with pytest.raises(core.UnsupportedFeatureError):
        engine.transcribe_file(audio, HTTP, timestamps="phrase")
    result = engine.transcribe_file(audio, HTTP)
    assert result["text"] == "hello there" and result["features"]["timestamps"] == "none"
    assert any("alignment" in w for w in result["warnings"])


def test_engine_passes_speakers_and_sounds_to_gemini(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.touch()
    seen = {}

    class FakeGemini:
        def speech_to_text(self, path, **kwargs):
            seen.update(kwargs)
            return {"text": "Speaker 1: hi", "no_speech": False}
    monkeypatch.setattr(engine, "get_duration", lambda path: 10.0)
    engine.transcribe_file(audio, GEMINI, gemini_api=FakeGemini(), sounds=False)
    assert seen["diarize"] is True and seen["sounds"] is False


def test_library_and_cli_defaults_match():
    signature = inspect.signature(engine.transcribe_file).parameters
    args = runner.build_parser().parse_args(["a.wav"])
    for name in ("speakers", "sounds", "timestamps"):
        assert signature[name].default is None and getattr(args, name) is None


@pytest.mark.parametrize("kind,flags,expected", [
    ("groq", ["--no-speakers"], {"speakers": False}), ("gemini", ["--speakers"], {"speakers": True}),
    ("gemini", ["--no-sounds"], {"sounds": False}), ("groq", ["--timestamps", "phrase"], {"timestamps": "phrase"}),
    ("gemini", ["--timestamps", "none"], {"timestamps": "none"}),
])
def test_cli_feature_flags_reach_the_engine(tmp_path, monkeypatch, capsys, kind, flags, expected):
    audio = tmp_path / "a.wav"
    audio.touch()
    config = tmp_path / "t.ini"
    config.write_text(f"[models]\nranking = w\n[model.w]\nkind = {kind}\n")
    seen = {}

    def fake(audio, spec, **kwargs):
        seen.update(kwargs)
        return {"text": "t", "warnings": []}
    monkeypatch.setattr(engine, "transcribe_file", fake)
    assert runner.main([str(audio), "--config", str(config), *flags]) == 0
    for key, value in expected.items():
        assert seen[key] == value


def test_cli_refuses_an_explicit_feature_no_ranked_model_supplies(tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.touch()
    config = tmp_path / "t.ini"
    config.write_text("[models]\nranking = w\n[model.w]\nkind = groq\n")
    with pytest.raises(SystemExit):
        runner.main([str(audio), "--config", str(config), "--speakers"])
    assert "cannot supply speaker labels" in capsys.readouterr().err


def test_word_alignment_is_grouped_into_phrases():
    # Seen live: the Parakeet server returns "words", not "segments".
    words = [{"word": "Как", "start": 0, "end": 0.16}, {"word": "только", "start": 0.16, "end": 0.32},
             {"word": "есть", "start": 0.8, "end": 0.96}, {"word": "какие-то", "start": 4, "end": 4.72},
             {"word": "мир.", "start": 7.12, "end": 7.36}, {"word": "выбирать", "start": 7.44, "end": 7.76}]
    assert engine._segments(None, words) == [
        {"start": 0.0, "end": 0.96, "text": "Как только есть"},
        {"start": 4.0, "end": 4.72, "text": "какие-то"},          # 2.4 s pause follows
        {"start": 7.12, "end": 7.36, "text": "мир."},             # sentence end closes the phrase
        {"start": 7.44, "end": 7.76, "text": "выбирать"}]


def test_http_word_alignment_gives_timestamps(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.touch()
    monkeypatch.setattr(engine, "get_duration", lambda path: 10.0)
    monkeypatch.setattr(engine, "assess_audio", lambda path: {"problems": []})
    from multimodal_transcription_core import http_client
    monkeypatch.setattr(http_client, "transcribe", lambda *a, **kw: (
        {"text": "Как только.", "words": [{"word": "Как", "start": 0, "end": 0.2},
                                          {"word": "только.", "start": 0.2, "end": 0.5}]}, b"", 200, 0.1))
    result = engine.transcribe_file(audio, HTTP)
    assert result["text"] == "[0:00:00] Как только." and result["features"]["timestamps"] == "major"
