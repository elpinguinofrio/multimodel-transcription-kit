"""Public transcribe functions use the caller's limited client end to end."""
from unittest.mock import MagicMock, patch

import pytest

import multimodal_transcription_core as core
from multimodal_transcription_core import chunked_transcriber, transcriber


def test_package_exports_limiter_types():
    for name in ("RateLimiter", "Limits", "QuotaExhausted", "RateLimitTimeout", "RateLimitError"):
        assert hasattr(core, name)


def test_transcribe_uses_the_given_client(tmp_path):
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    api = MagicMock()
    api.speech_to_text.return_value = {"text": "ok"}
    assert transcriber.transcribe(audio, api=api)["text"] == "ok"
    api.speech_to_text.assert_called_once()


def test_transcribe_lets_quota_errors_through_unwrapped(tmp_path):
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    api = MagicMock()
    api.speech_to_text.side_effect = core.QuotaExhausted("m", 0.0, "test")
    with pytest.raises(core.QuotaExhausted):
        transcriber.transcribe(audio, api=api)


def test_merge_goes_through_generate_text():
    api = MagicMock()
    api.generate_text.return_value = "merged"
    assert chunked_transcriber.ChunkedMergeAPI(api).generate_text("p") == "merged"
    assert api.generate_text.call_args.kwargs["model"] == chunked_transcriber.MERGE_MODEL


def test_merge_failure_returns_empty_but_quota_propagates():
    api = MagicMock()
    api.generate_text.side_effect = core.GeminiAPIError("boom")
    assert chunked_transcriber.ChunkedMergeAPI(api).generate_text("p") == ""
    api.generate_text.side_effect = core.QuotaExhausted("m", 0.0, "test")
    with pytest.raises(core.QuotaExhausted):
        chunked_transcriber.ChunkedMergeAPI(api).generate_text("p")


def test_quota_exhausted_aborts_chunked_transcription(tmp_path):
    audio = tmp_path / "long.m4a"
    audio.touch()
    chunks = [MagicMock(path=f"/tmp/c{i}.m4a", start_time=i * 1800, end_time=(i + 1) * 1800)
              for i in range(3)]
    api = MagicMock()
    with patch.object(chunked_transcriber, "get_duration", return_value=5400.0), \
         patch.object(chunked_transcriber, "split_audio", return_value=chunks), \
         patch.object(chunked_transcriber, "cleanup_chunks") as cleanup, \
         patch.object(chunked_transcriber, "_transcribe_chunk", side_effect=[
             {"text": "part one", "success": True},
             core.QuotaExhausted("m", 0.0, "test"),
             {"text": "part three", "success": True}]):
        with pytest.raises(core.QuotaExhausted):
            chunked_transcriber.transcribe_large(audio, api=api)
    cleanup.assert_called_once_with(chunks)


def test_short_file_passes_client_to_standard_path(tmp_path):
    audio = tmp_path / "short.m4a"
    audio.touch()
    api = MagicMock()
    with patch.object(chunked_transcriber, "get_duration", return_value=60.0), \
         patch.object(chunked_transcriber, "_standard_transcribe", return_value={"text": "t"}) as std:
        chunked_transcriber.transcribe_large(audio, api=api)
    assert std.call_args.kwargs["api"] is api


def test_failed_boundary_merge_keeps_every_word():
    from multimodal_transcription_core.chunk_merger import merge_adjacent_chunks
    a = "alpha " * 100 + "TAIL_A_END"
    b = "HEAD_B_START " + "beta " * 100
    api = MagicMock()
    api.generate_text.return_value = ""
    assert merge_adjacent_chunks(a, b, api) == a + " " + b


def test_quota_during_merge_still_cleans_up_chunks(tmp_path):
    audio = tmp_path / "long.m4a"
    audio.touch()
    chunks = [MagicMock(path=f"/tmp/c{i}.m4a", start_time=i * 1800, end_time=(i + 1) * 1800)
              for i in range(2)]
    with patch.object(chunked_transcriber, "get_duration", return_value=3600.0), \
         patch.object(chunked_transcriber, "split_audio", return_value=chunks), \
         patch.object(chunked_transcriber, "cleanup_chunks") as cleanup, \
         patch.object(chunked_transcriber, "_transcribe_chunk", return_value={"text": "t", "success": True}), \
         patch.object(chunked_transcriber, "merge_all_chunks",
                      side_effect=core.QuotaExhausted("m", 0.0, "test")):
        with pytest.raises(core.QuotaExhausted):
            chunked_transcriber.transcribe_large(audio, api=MagicMock())
    cleanup.assert_called_once_with(chunks)


def test_transcribe_large_model_is_limited_for_chunks_and_merge(tmp_path):
    """Every request of a long file counts under the model the caller chose."""
    audio = tmp_path / "long.m4a"
    audio.touch()
    chunks = [MagicMock(path=str(tmp_path / f"c{i}.m4a"), start_time=i * 1800, end_time=(i + 1) * 1800)
              for i in range(2)]
    for c in chunks:
        open(c.path, "wb").close()
    rl = core.RateLimiter({"gemini-3.5-flash-lite": core.Limits(rpd=500)})
    api = core.GeminiAPI(rate_limiter=rl)
    ok = MagicMock(status_code=200, headers={})
    ok.json.return_value = {"candidates": [{"content": {"parts": [{"text": "words " * 200}]}}]}
    with patch.object(chunked_transcriber, "get_duration", return_value=3600.0), \
         patch.object(chunked_transcriber, "split_audio", return_value=chunks), \
         patch.object(chunked_transcriber, "cleanup_chunks"), \
         patch.object(api, "_get_audio_duration", return_value=1800.0, create=True), \
         patch("requests.post", return_value=ok):
        result = chunked_transcriber.transcribe_large(audio, api=api, model="gemini-3.5-flash-lite")
    assert result["model"] == "gemini-3.5-flash-lite"
    assert rl.usage("gemini-3.5-flash-lite")["today"] == 3   # 2 chunks + 1 merge


def test_transcribe_large_short_file_uses_chosen_model(tmp_path):
    audio = tmp_path / "short.m4a"
    audio.touch()
    with patch.object(chunked_transcriber, "get_duration", return_value=60.0), \
         patch.object(chunked_transcriber, "_standard_transcribe", return_value={"text": "t"}) as std:
        chunked_transcriber.transcribe_large(audio, api=MagicMock(), model="gemini-3.5-flash-lite")
    assert std.call_args.kwargs["model"] == "gemini-3.5-flash-lite"
