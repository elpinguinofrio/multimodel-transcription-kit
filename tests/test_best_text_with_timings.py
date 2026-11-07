"""Owner decision (2026-11-07): timestamps never demote the text to a weaker model.

The best ranked model writes the text; when it cannot align words, the best
timing-capable model is run only for word times and the text is aligned onto them.
"""
from multimodel_transcription_kit import selection
from multimodel_transcription_kit.alignment import align_words
from multimodel_transcription_kit.engine import ModelSpec
from multimodel_transcription_kit.errors import NoModelAvailableError

import pytest

GEMINI = ModelSpec(name="gemini", kind="gemini", model="gemini-3.8-flash")
WHISPER = ModelSpec(name="whisper", kind="groq", model="whisper-large-v3")

TIMED = [{"word": "Привет", "start": 0.0, "end": 0.5}, {"word": "ChachaGPT", "start": 0.6, "end": 1.2},
         {"word": "и", "start": 1.3, "end": 1.4}, {"word": "клоу", "start": 1.5, "end": 2.0}]


def fake(calls):
    def transcribe(audio, spec, **kw):
        calls.append((spec.name, kw.get("timestamps"), kw.get("keep_raw")))
        if spec.kind == "gemini":
            return {"text": "Привет, ChatGPT и Claude.", "plain_text": "Привет, ChatGPT и Claude.",
                    "name": spec.name, "warnings": [], "features": {"timestamps": "none"}}
        return {"text": "[00:00] Привет ChachaGPT и клоу", "plain_text": "Привет ChachaGPT и клоу",
                "name": spec.name, "warnings": [],
                "parts": [{"part": 1, "start": 10.0, "end": 12.0, "failed": False,
                           "raw_response": {"words": TIMED}}]}
    return transcribe


def test_best_model_writes_text_and_timing_model_only_supplies_times():
    calls = []
    result = selection.transcribe_best("x.wav", [GEMINI, WHISPER], probe=lambda s: True,
                                       transcribe=fake(calls), timestamps="phrase")
    assert calls == [("gemini", "none", None), ("whisper", "phrase", True)]
    assert result["selection"]["chosen"] == "gemini"
    assert result["alignment"]["text_model"] == "gemini" and result["alignment"]["timing_model"] == "whisper"
    assert [w["word"] for w in result["words"]] == ["Привет,", "ChatGPT", "и", "Claude."]
    assert [(w["start"], w["end"]) for w in result["words"]] == [(10.0, 10.5), (10.6, 11.2), (11.3, 11.4), (11.5, 12.0)]
    assert result["text"].startswith("[0:00:10] Привет, ChatGPT и Claude.")
    assert result["features"]["timestamps"] == "phrase"


def test_timing_capable_best_model_is_used_alone():
    calls = []
    selection.transcribe_best("x.wav", [WHISPER, GEMINI], probe=lambda s: True,
                              transcribe=fake(calls), timestamps="phrase")
    assert [c[0] for c in calls] == ["whisper"]


def test_no_timing_model_is_an_error_not_a_silent_downgrade():
    with pytest.raises(NoModelAvailableError, match="timing"):
        selection.transcribe_best("x.wav", [GEMINI, WHISPER], probe=lambda s: s.name == "gemini",
                                  transcribe=fake([]), timestamps="phrase")


def test_alignment_interpolates_words_the_timing_model_missed():
    timed = [{"word": "один", "start": 0.0, "end": 1.0}, {"word": "четыре", "start": 3.0, "end": 4.0}]
    words = align_words("один два три четыре", timed)
    assert [w["word"] for w in words] == ["один", "два", "три", "четыре"]
    assert words[1]["start"] == 1.0 and words[2]["end"] == 3.0
    assert words[1]["start"] < words[1]["end"] <= words[2]["start"]
    assert [w["aligned"] for w in words] == ["matched", "interpolated", "interpolated", "matched"]


def test_alignment_drops_speaker_labels_sound_notes_and_bare_punctuation():
    timed = [{"word": "как", "start": 0.0, "end": 0.2}, {"word": "дела", "start": 0.2, "end": 0.6}]
    words = align_words("Speaker 1: так, как *dog barking* , дела?", timed)
    assert [w["word"] for w in words] == ["так,", "как", "дела?"]


def test_aligned_words_are_ordered_and_never_zero_length():
    timed = [{"word": "как", "start": 0.0, "end": 0.4}, {"word": "дела", "start": 0.4, "end": 0.8}]
    for text in ("так вот как дела", "как дела ну вот", "как ну вот дела"):
        words = align_words(text, timed)
        assert len(words) == 4, text
        assert all(w["end"] > w["start"] for w in words), (text, words)
        assert all(a["end"] <= b["start"] for a, b in zip(words, words[1:])), (text, words)
        assert words[0]["start"] >= 0.0 and words[-1]["end"] <= 0.8, (text, words)


def test_overlapping_timing_words_still_give_ordered_output():
    timed = [{"word": "раз", "start": 0.0, "end": 1.0}, {"word": "два", "start": 0.5, "end": 1.2},
             {"word": "три", "start": 1.2, "end": 1.2}, {"word": "пять", "start": 1.1, "end": 2.0}]
    words = align_words("раз два три четыре пять шесть семь", timed)
    assert len(words) == 7
    assert all(w["end"] > w["start"] for w in words), words
    assert all(a["end"] <= b["start"] for a, b in zip(words, words[1:])), words


def test_angle_bracket_markers_are_not_subtitle_words():
    timed = [{"word": "ну", "start": 0.0, "end": 0.3}, {"word": "да", "start": 0.3, "end": 0.6}]
    assert [w["word"] for w in align_words("ну <неразборчиво> да <pant>", timed)] == ["ну", "да"]


PARAKEET = ModelSpec(name="parakeet", kind="http", model="parakeet", endpoint="http://127.0.0.1:9/v1")


def test_timing_order_is_its_own_list_not_the_text_ranking():
    calls = []
    result = selection.transcribe_best("x.wav", [GEMINI, WHISPER, PARAKEET], probe=lambda s: True,
                                       transcribe=fake(calls), timestamps="phrase",
                                       timing_specs=[PARAKEET, WHISPER])
    assert [c[0] for c in calls] == ["gemini", "parakeet"]
    assert result["alignment"]["timing_model"] == "parakeet"


def test_timing_order_falls_back_when_the_first_timing_model_fails():
    calls, inner = [], fake([])

    def transcribe(audio, spec, **kw):
        calls.append(spec.name)
        if spec.name == "parakeet":
            raise selection.TranscriptionError("down")
        return inner(audio, spec, **kw)
    result = selection.transcribe_best("x.wav", [GEMINI], probe=lambda s: True, transcribe=transcribe,
                                       timestamps="phrase", timing_specs=[PARAKEET, WHISPER])
    assert calls == ["gemini", "parakeet", "whisper"]
    assert result["alignment"]["timing_model"] == "whisper"
    assert result["alignment"]["timing_selection"]["attempts"][0]["name"] == "parakeet"


def test_cli_reads_timing_ranking_from_config(tmp_path, monkeypatch):
    from multimodel_transcription_kit import runner
    audio = tmp_path / "talk.wav"
    audio.write_bytes(b"x")
    config = tmp_path / "t.ini"
    config.write_text("[models]\nranking = gemini\ntiming_ranking = parakeet, whisper\n\n"
                      "[model.gemini]\nkind = gemini\n\n[model.whisper]\nkind = groq\nmodel = whisper-large-v3\n\n"
                      "[model.parakeet]\nkind = http\nmodel = parakeet\nendpoint = http://127.0.0.1:9/v1\n")
    monkeypatch.setattr(runner, "is_reachable", lambda name, options: True)
    calls, inner = [], fake([])

    def transcribe_with_model(name, options, audio_path, language, **kw):
        calls.append(name)
        return inner(audio_path, runner.model_spec(None, name, options), **kw)
    monkeypatch.setattr(runner, "transcribe_with_model", transcribe_with_model)
    assert runner.main([str(audio), "--config", str(config), "--timestamps", "phrase"]) == 0
    assert calls == ["gemini", "parakeet"]


def test_native_timing_model_also_fills_result_words():
    # Callers read only result["text"] and result["words"], whichever model ran.
    calls = []
    result = selection.transcribe_best("x.wav", [WHISPER, GEMINI], probe=lambda s: True,
                                       transcribe=fake(calls), timestamps="phrase")
    assert calls == [("whisper", "phrase", True)]
    assert [(w["word"], w["start"], w["end"]) for w in result["words"]] == [
        ("Привет", 10.0, 10.5), ("ChachaGPT", 10.6, 11.2), ("и", 11.3, 11.4), ("клоу", 11.5, 12.0)]


def test_no_timestamps_requested_leaves_words_unset():
    result = selection.transcribe_best("x.wav", [WHISPER], probe=lambda s: True, transcribe=fake([]))
    assert "words" not in result
