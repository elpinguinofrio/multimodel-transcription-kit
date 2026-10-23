"""Regression coverage for the two Gemini transcription response shapes."""

from unittest.mock import Mock

import pytest

from multimodel_transcription_kit import GeminiAPI


@pytest.mark.parametrize("parts,expected", [
    ([{"audioTranscription": {"text": "spoken words"}}], "spoken words"),
    ([{"text": "ordinary text"}], "ordinary text"),
    ([{"thought": True, "text": "reasoning"},
      {"audioTranscription": {"text": "spoken words"}}], "spoken words"),
    ([{"text": "first "}, {"audioTranscription": {"text": "second"}}], "first second"),
])
def test_reads_transcription_parts(monkeypatch, parts, expected):
    api = GeminiAPI(throttle_rate=0)
    monkeypatch.setattr(api, "_rate_limit", lambda: None)
    response = Mock(status_code=200)
    response.json.return_value = {"candidates": [{"content": {"parts": parts}}]}
    monkeypatch.setattr(api, "_post_with_retry", lambda *a, **kw: response)
    result = api.speech_to_text(b"synthetic", model="gemini-3.5-transcribe")
    assert result["text"] == expected


def test_missing_transcription_raises_with_diagnostic(monkeypatch):
    from multimodel_transcription_kit import ResponseError
    api = GeminiAPI(throttle_rate=0)
    monkeypatch.setattr(api, "_rate_limit", lambda: None)
    response = Mock(status_code=200)
    response.json.return_value = {"candidates": [{"finishReason": "SAFETY"}]}
    monkeypatch.setattr(api, "_post_with_retry", lambda *a, **kw: response)
    with pytest.raises(ResponseError, match="finishReason=SAFETY"):
        api.speech_to_text(b"synthetic")
