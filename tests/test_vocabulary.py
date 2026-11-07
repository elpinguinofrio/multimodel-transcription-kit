"""Vocabulary: term hints sent to models and corrections of known mishearings."""
import pytest

from multimodel_transcription_kit import engine, runner
from multimodel_transcription_kit.engine import ModelSpec
from multimodel_transcription_kit.vocabulary import correct, parse_corrections, parse_terms

FIXES = parse_corrections("веб-кодинг -> вайбкодинг\nвейп козинг -> вайбкодинг; вебкодил -> вайбкодил")


def test_parse_terms_and_corrections():
    assert parse_terms("вайбкодинг, Claude,\n Codex ,") == ["вайбкодинг", "Claude", "Codex"]
    assert FIXES == [("веб-кодинг", "вайбкодинг"), ("вейп козинг", "вайбкодинг"), ("вебкодил", "вайбкодил")]
    with pytest.raises(ValueError, match="->"):
        parse_corrections("no arrow here")


@pytest.mark.parametrize("heard,fixed", [
    ("это для веб-кодинга, то", "это для вайбкодинга, то"),          # Russian ending kept
    ("Веб-кодинг рулит", "Вайбкодинг рулит"),                         # capital kept
    ("для вейп козинга", "для вайбкодинга"),
    ("для вейп-козинга", "для вайбкодинга"),                          # space, hyphen or nothing
    ("для вейпкозинга", "для вайбкодинга"),
    ("ничего не вебкодил", "ничего не вайбкодил"),
])
def test_corrections_fix_known_mishearings(heard, fixed):
    text, applied = correct(heard, FIXES)
    assert text == fixed and len(applied) == 1


def test_corrections_never_touch_the_middle_of_a_word():
    assert correct("суперверб-кодинг", parse_corrections("верб-кодинг -> x"))[0] == "суперверб-кодинг"


def fake_request(seen, text="ничего не вебкодил для веб-кодинга"):
    def request(spec, path, settings, clients):
        seen.append((spec, dict(settings)))
        return {"text": text, "segments": [{"start": 0.0, "end": 2.0, "text": text}]}
    return request


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"RIFF" + b"\0" * 64)
    return path


def test_engine_applies_corrections_to_text_and_segments(monkeypatch, audio):
    seen = []
    monkeypatch.setattr(engine, "_request", fake_request(seen))
    spec = ModelSpec(name="w", kind="groq", model="whisper-large-v3")
    result = engine.transcribe_file(audio, spec, corrections=FIXES, enhance=False, timestamps="phrase")
    assert result["plain_text"] == "ничего не вайбкодил для вайбкодинга"
    assert result["segments"][0]["text"] == "ничего не вайбкодил для вайбкодинга"
    assert "вайбкодинга" in result["text"] and len(result["corrections"]) == 2


@pytest.mark.parametrize("spec,prompt,expected", [
    (ModelSpec(name="g", kind="gemini", model="gemini-3.6-flash"), None, "spell them exactly"),
    (ModelSpec(name="g", kind="gemini", model="gemini-3.6-flash"), "Verbatim.", "Verbatim. "),
    (ModelSpec(name="w", kind="groq", model="whisper-large-v3"), None, "вайбкодинг, Claude."),
])
def test_engine_sends_term_hints(monkeypatch, audio, spec, prompt, expected):
    seen = []
    monkeypatch.setattr(engine, "_request", fake_request(seen))
    engine.transcribe_file(audio, spec, prompt=prompt, terms=["вайбкодинг", "Claude"], enhance=False)
    hint = seen[0][1]["hint"]
    sent = (seen[0][1]["prompt"] or "") + " " + hint
    assert expected in sent and "вайбкодинг" in hint


def test_live_models_get_corrections_but_warn_about_hints(monkeypatch, audio):
    monkeypatch.setattr(engine, "_request", fake_request([]))
    spec = ModelSpec(name="l", kind="gemini", model="gemini-3.5-transcribe-live")
    result = engine.transcribe_file(audio, spec, terms=["вайбкодинг"], corrections=FIXES, enhance=False)
    assert "вайбкодинга" in result["text"]
    assert any("hint" in w for w in result["warnings"])


def test_cli_reads_vocabulary_from_config_and_flags(tmp_path, monkeypatch, audio):
    config = tmp_path / "t.ini"
    config.write_text("[models]\nranking = w\n\n[model.w]\nkind = groq\nmodel = whisper-large-v3\n\n"
                      "[vocabulary]\nterms = вайбкодинг, Claude\ncorrections =\n  веб-кодинг -> вайбкодинг\n")
    monkeypatch.setattr(runner, "is_reachable", lambda name, options: True)
    seen = {}

    def transcribe_with_model(name, options, audio_path, language, **kw):
        seen.update(kw)
        return {"text": "x", "name": name, "warnings": []}
    monkeypatch.setattr(runner, "transcribe_with_model", transcribe_with_model)
    assert runner.main([str(audio), "--config", str(config), "--term", "OpenClaw",
                        "--correction", "клоу -> Claude"]) == 0
    assert seen["terms"] == ["вайбкодинг", "Claude", "OpenClaw"]
    assert seen["corrections"] == [("веб-кодинг", "вайбкодинг"), ("клоу", "Claude")]
