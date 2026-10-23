"""Failures raise typed errors; no-speech and unknown usage are explicit in every result."""

from unittest.mock import MagicMock, Mock, patch

import pytest

import multimodel_transcription_kit as core
from multimodel_transcription_kit import GeminiAPI, GroqAPI, STTModel
from multimodel_transcription_kit import chunked_transcriber


def gemini_returning(monkeypatch, body):
    api = GeminiAPI(throttle_rate=0)
    monkeypatch.setattr(api, "_rate_limit", lambda: None)
    response = Mock(status_code=200)
    response.json.return_value = body
    monkeypatch.setattr(api, "_post_with_retry", lambda *a, **kw: response)
    return api


def test_errors_share_one_base_and_keep_runtime_error_compatibility():
    assert issubclass(core.GeminiAPIError, core.TranscriptionError)
    assert issubclass(core.GroqAPIError, core.TranscriptionError)
    assert issubclass(core.ResponseError, core.TranscriptionError)
    assert issubclass(core.IncompleteTranscriptError, core.TranscriptionError)
    assert issubclass(core.TranscriptionError, RuntimeError)


@pytest.mark.parametrize("body,detail", [
    ({"candidates": [{"finishReason": "SAFETY"}]}, "finishReason=SAFETY"),
    ({"candidates": [{"finishReason": "PROHIBITED_CONTENT", "content": {}}]}, "PROHIBITED_CONTENT"),
    ({"promptFeedback": {"blockReason": "OTHER"}}, "PROMPT BLOCKED: OTHER"),
    ({"unexpected": True}, "No candidates"),
])
def test_gemini_blocked_or_malformed_response_raises(monkeypatch, body, detail):
    api = gemini_returning(monkeypatch, body)
    with pytest.raises(core.ResponseError, match=detail) as caught:
        api.speech_to_text(b"synthetic")
    assert isinstance(caught.value, core.GeminiAPIError)


def test_gemini_truncated_transcript_raises_with_partial_text(monkeypatch):
    api = gemini_returning(monkeypatch, {"candidates": [{
        "finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "first half"}]}}]})
    with pytest.raises(core.ResponseError, match="MAX_TOKENS") as caught:
        api.speech_to_text(b"synthetic")
    assert caught.value.partial_text == "first half"
    assert caught.value.finish_reason == "MAX_TOKENS"


def test_gemini_repetition_loop_raises(monkeypatch):
    api = gemini_returning(monkeypatch, {"candidates": [{
        "finishReason": "STOP", "content": {"parts": [{"text": "да да да. " * 40}]}}]})
    with pytest.raises(core.ResponseError, match="repetition"):
        api.speech_to_text(b"synthetic")


@pytest.mark.parametrize("text", ["[SILENCE]", " [silence]\n", ""])
def test_gemini_silence_is_explicit_no_speech(monkeypatch, text):
    api = gemini_returning(monkeypatch, {"candidates": [{
        "finishReason": "STOP", "content": {"parts": [{"text": text}]}}]})
    result = api.speech_to_text(b"synthetic")
    assert result["success"] is True
    assert result["no_speech"] is True


def test_gemini_speech_is_not_no_speech(monkeypatch):
    api = gemini_returning(monkeypatch, {"candidates": [{
        "finishReason": "STOP", "content": {"parts": [{"text": "hello there"}]}}]})
    result = api.speech_to_text(b"synthetic")
    assert result["no_speech"] is False and result["provider"] == "gemini"


def test_gemini_missing_usage_stays_unknown(monkeypatch):
    api = gemini_returning(monkeypatch, {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
    result = api.speech_to_text(b"synthetic")
    assert result["input_tokens"] is None and result["output_tokens"] is None


def test_gemini_reported_usage_is_kept(monkeypatch):
    api = gemini_returning(monkeypatch, {
        "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3}})
    result = api.speech_to_text(b"synthetic")
    assert (result["input_tokens"], result["output_tokens"]) == (12, 3)


def groq_returning(monkeypatch, body):
    api = GroqAPI(throttle_rate=0)
    response = Mock(status_code=200)
    response.json.return_value = body
    from multimodel_transcription_kit import groq_client
    monkeypatch.setattr(groq_client, "send_with_limits", lambda *a, **kw: response)
    return api


def test_groq_result_has_the_common_shape(monkeypatch):
    api = groq_returning(monkeypatch, {"text": " hello "})
    result = api.speech_to_text(b"x" * 64, model=STTModel.WHISPER_LARGE_V3)
    assert result["input_tokens"] is None and result["output_tokens"] is None
    assert result["model"] == "whisper-large-v3" and result["provider"] == "groq"
    assert result["no_speech"] is False and result["success"] is True


def test_groq_empty_text_is_explicit_no_speech(monkeypatch):
    api = groq_returning(monkeypatch, {"text": "  "})
    result = api.speech_to_text(b"x" * 64, model=STTModel.WHISPER_LARGE_V3)
    assert result["no_speech"] is True


def test_groq_text_without_string_raises(monkeypatch):
    api = groq_returning(monkeypatch, {"error": "nope"})
    with pytest.raises(core.ResponseError):
        api.speech_to_text(b"x" * 64, model=STTModel.WHISPER_LARGE_V3)


# ---------------------------------------------------------------- chunked

def run_chunked(tmp_path, chunk_results, merged="merged"):
    from multimodel_transcription_kit import engine
    audio = tmp_path / "long.m4a"
    audio.touch()
    chunks = [MagicMock(path=f"/tmp/c{i}.m4a", start_time=i * 1800, end_time=(i + 1) * 1800 + 10)
              for i in range(len(chunk_results))]
    rebuilt = {"text": merged, "segments": None, "warnings": [], "boundaries": []}
    with patch.object(engine, "get_duration", return_value=1800.0 * len(chunks)), \
            patch.object(engine, "split_audio", return_value=chunks), \
            patch.object(engine, "_request", side_effect=chunk_results), \
            patch.object(engine, "reconstruct", return_value=rebuilt) as merge:
        return chunked_transcriber.transcribe_large(str(audio)), merge


def test_failed_chunk_beside_speech_raises_incomplete_with_partial(tmp_path):
    with pytest.raises(core.IncompleteTranscriptError) as caught:
        run_chunked(tmp_path, [{"text": "speech", "success": True},
                               core.ResponseError("blocked", finish_reason="SAFETY")])
    assert caught.value.failed_parts == [2]
    assert caught.value.result["text"] == "merged"
    assert caught.value.result["success"] is False


def test_parse_error_text_from_a_legacy_client_counts_as_failed(tmp_path):
    with pytest.raises(core.IncompleteTranscriptError):
        run_chunked(tmp_path, [{"text": "[PARSE_ERROR] finishReason=SAFETY"}, {"text": "speech"}])


def test_all_silent_chunks_report_no_speech(tmp_path):
    result, _ = run_chunked(tmp_path, [{"text": "[SILENCE]"}, {"text": ""}])
    assert result["success"] is True and result["no_speech"] is True


def test_chunk_usage_is_unknown_when_any_chunk_lacks_it(tmp_path):
    result, _ = run_chunked(tmp_path, [{"text": "a", "input_tokens": 5, "output_tokens": 1},
                                       {"text": "b", "input_tokens": None, "output_tokens": 2}])
    assert result["input_tokens"] is None and result["output_tokens"] == 3


@pytest.mark.parametrize("text", ["Speaker 1: [SILENCE]", "Speaker 1: [SILENCE]\nSpeaker 2: [silence]"])
def test_labelled_silence_is_no_speech(text):
    # Seen live: gemini-3.5-flash-lite with diarization answered "Speaker 1: [SILENCE]".
    from multimodel_transcription_kit.transcription_types import is_no_speech
    assert is_no_speech(text)
    assert not is_no_speech("Speaker 1: [SILENCE] hello")
