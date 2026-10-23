"""Public transcribe functions use the caller's limited client end to end."""
from unittest.mock import MagicMock, patch

import pytest

import multimodel_transcription_kit as core
from multimodel_transcription_kit import chunked_transcriber, transcriber
from multimodel_transcription_kit import engine


def engine_seams(tmp_path, duration, chunks=2):
    """Patch the engine's duration and splitting; chunk files land in its work dir."""
    made = []

    def split(path, chunk_sec, overlap_sec, output_dir):
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out = []
        for i in range(chunks):
            chunk = Path(output_dir) / f"c{i}.m4a"
            chunk.touch()
            made.append(chunk)
            out.append(MagicMock(path=str(chunk), start_time=i * 1800.0, end_time=(i + 1) * 1800.0 + 10))
        return out
    return made, patch.object(engine, "get_duration", return_value=duration), \
        patch.object(engine, "split_audio", side_effect=split)


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
    made, duration, split = engine_seams(tmp_path, 5400.0, chunks=3)
    request = MagicMock(side_effect=[{"text": "part one", "success": True},
                                     core.QuotaExhausted("m", 0.0, "test"),
                                     {"text": "part three", "success": True}])
    with duration, split, patch.object(engine, "_request", request):
        with pytest.raises(core.QuotaExhausted):
            chunked_transcriber.transcribe_large(audio, api=MagicMock())
    assert request.call_count == 2                       # no request after the quota stop
    assert made and not any(f.exists() for f in made)   # chunk files cleaned up


def test_short_file_passes_client_to_standard_path(tmp_path):
    audio = tmp_path / "short.m4a"
    audio.touch()
    api = MagicMock()
    request = MagicMock(return_value={"text": "t"})
    with patch.object(engine, "get_duration", return_value=60.0), patch.object(engine, "_request", request):
        chunked_transcriber.transcribe_large(audio, api=api)
    clients = request.call_args[0][3]
    assert clients.gemini is api


def test_failed_boundary_merge_keeps_every_word():
    from multimodel_transcription_kit.chunk_merger import merge_adjacent_chunks
    a = "alpha " * 100 + "TAIL_A_END"
    b = "HEAD_B_START " + "beta " * 100
    api = MagicMock()
    api.generate_text.return_value = ""
    assert merge_adjacent_chunks(a, b, api) == a + " " + b


def test_quota_during_merge_still_cleans_up_chunks(tmp_path):
    audio = tmp_path / "long.m4a"
    audio.touch()
    made, duration, split = engine_seams(tmp_path, 3600.0)
    with duration, split, \
         patch.object(engine, "_request", return_value={"text": "t", "success": True}), \
         patch.object(engine, "reconstruct", side_effect=core.QuotaExhausted("m", 0.0, "test")):
        with pytest.raises(core.QuotaExhausted):
            chunked_transcriber.transcribe_large(audio, api=MagicMock())
    assert made and not any(f.exists() for f in made)


def test_transcribe_large_model_is_limited_for_chunks_and_merge(tmp_path):
    """Every request of a long file counts under the model the caller chose."""
    audio = tmp_path / "long.m4a"
    audio.touch()
    rl = core.RateLimiter({"gemini-3.5-flash-lite": core.Limits(rpd=500)})
    api = core.GeminiAPI(rate_limiter=rl)

    def reply(text):
        response = MagicMock(status_code=200, headers={})
        response.json.return_value = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
        return response
    first = " ".join(f"first{i}" for i in range(100))
    second = " ".join(f"second{i}" for i in range(100))
    # No words shared at the boundary, so the merge model is asked once.
    responses = [reply(first), reply(second), reply(first[-200:] + " " + second[:200])]
    made, duration, split = engine_seams(tmp_path, 3600.0)
    with duration, split, \
         patch.object(api, "_get_audio_duration", return_value=1800.0, create=True), \
         patch("requests.post", side_effect=responses):
        result = chunked_transcriber.transcribe_large(audio, api=api, model="gemini-3.5-flash-lite")
    assert result["model"] == "gemini-3.5-flash-lite"
    assert result["boundaries"][0]["method"] == "model"
    assert rl.usage("gemini-3.5-flash-lite")["today"] == 3   # 2 chunks + 1 merge


def test_transcribe_large_short_file_uses_chosen_model(tmp_path):
    audio = tmp_path / "short.m4a"
    audio.touch()
    request = MagicMock(return_value={"text": "t"})
    with patch.object(engine, "get_duration", return_value=60.0), patch.object(engine, "_request", request):
        chunked_transcriber.transcribe_large(audio, api=MagicMock(), model="gemini-3.5-flash-lite")
    assert request.call_args[0][0].model == "gemini-3.5-flash-lite"


