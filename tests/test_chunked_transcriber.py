"""
Tests for Chunked Transcriber - handles large audio files.

TDD Red Phase: These tests should FAIL until implementation is complete.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib.util


def _load_module(name: str):
    from importlib import import_module
    return import_module(f"multimodal_transcription_core.{name}")


# =============================================================================
# AUDIO CHUNKER TESTS
# =============================================================================

class TestAudioChunker:
    """Tests for audio_chunker.py - ffmpeg-based audio splitting."""

    def test_split_audio_creates_chunks(self, tmp_path):
        """Split a 1-hour file into 30-min chunks."""
        chunker = _load_module("audio_chunker")

        # Create a mock audio file (we'll mock ffmpeg)
        audio_file = tmp_path / "test_1hour.m4a"
        audio_file.touch()

        with patch.object(chunker, "_run_ffmpeg") as mock_ffmpeg:
            # Mock ffprobe returning 3600 seconds (1 hour)
            with patch.object(chunker, "get_duration", return_value=3600.0):
                chunks = chunker.split_audio(str(audio_file), chunk_sec=1800, overlap_sec=10)

        # 1 hour with 30-min chunks = 2 chunks
        # Overlap achieved by EXTENDING end, not shifting start
        assert len(chunks) == 2
        assert chunks[0].start_time == 0
        assert chunks[0].end_time == 1810  # 30min + 10s overlap
        assert chunks[1].start_time == 1800  # Starts at round interval
        assert chunks[1].end_time == 3600  # Last chunk extends to end

    def test_split_audio_overlap_duration(self, tmp_path):
        """Verify 10-second overlap between adjacent chunks."""
        chunker = _load_module("audio_chunker")

        audio_file = tmp_path / "test.m4a"
        audio_file.touch()

        with patch.object(chunker, "_run_ffmpeg"):
            with patch.object(chunker, "get_duration", return_value=5400.0):  # 1.5 hours
                chunks = chunker.split_audio(str(audio_file), chunk_sec=1800, overlap_sec=10)

        # Check overlap: chunk N ends, chunk N+1 starts 10s earlier
        for i in range(len(chunks) - 1):
            overlap = chunks[i].end_time - chunks[i + 1].start_time
            assert overlap == 10, f"Expected 10s overlap, got {overlap}s"

    def test_split_audio_returns_audiochunk_dataclass(self, tmp_path):
        """Chunks should be AudioChunk dataclass with path and timestamps."""
        chunker = _load_module("audio_chunker")

        audio_file = tmp_path / "test.m4a"
        audio_file.touch()

        with patch.object(chunker, "_run_ffmpeg"):
            with patch.object(chunker, "get_duration", return_value=3600.0):
                chunks = chunker.split_audio(str(audio_file))

        chunk = chunks[0]
        assert hasattr(chunk, "path"), "AudioChunk must have 'path' attribute"
        assert hasattr(chunk, "start_time"), "AudioChunk must have 'start_time' attribute"
        assert hasattr(chunk, "end_time"), "AudioChunk must have 'end_time' attribute"

    def test_split_audio_small_file_returns_single_chunk(self, tmp_path):
        """Files under threshold return as single chunk."""
        chunker = _load_module("audio_chunker")

        audio_file = tmp_path / "short.m4a"
        audio_file.touch()

        with patch.object(chunker, "_run_ffmpeg"):
            with patch.object(chunker, "get_duration", return_value=600.0):  # 10 minutes
                chunks = chunker.split_audio(str(audio_file), chunk_sec=1800)

        assert len(chunks) == 1
        assert chunks[0].start_time == 0
        assert chunks[0].end_time == 600


# =============================================================================
# CHUNK MERGER TESTS
# =============================================================================

class TestChunkMerger:
    """Tests for chunk_merger.py - LLM-based overlap deduplication."""

    def test_merge_uses_llm_for_boundary(self):
        """Merger should call LLM to combine overlapping regions."""
        merger = _load_module("chunk_merger")

        chunk_a = "The quick brown fox jumps over the lazy dog. The end of chunk A."
        chunk_b = "The end of chunk A. And this is the start of chunk B continues here."

        mock_api = MagicMock()
        mock_api.generate_text.return_value = "The end of chunk A. And this is the start of chunk B"

        result = merger.merge_adjacent_chunks(chunk_a, chunk_b, mock_api)

        # LLM should have been called
        mock_api.generate_text.assert_called_once()

    def test_merge_extracts_correct_windows(self):
        """Merger should extract last 500 chars of A and first 500 chars of B."""
        merger = _load_module("chunk_merger")

        # Create chunks longer than 500 chars
        chunk_a = "A" * 600 + "TAIL_A_END"
        chunk_b = "HEAD_B_START" + "B" * 600

        mock_api = MagicMock()
        mock_api.generate_text.return_value = "merged boundary"

        merger.merge_adjacent_chunks(chunk_a, chunk_b, mock_api)

        # Check the prompt contains the window regions
        call_args = mock_api.generate_text.call_args[0][0]
        assert "TAIL_A_END" in call_args, "Prompt should contain tail of chunk A"
        assert "HEAD_B_START" in call_args, "Prompt should contain head of chunk B"

    def test_merge_stitches_correctly(self):
        """Result should be: chunk_a[:-500] + merged_boundary + chunk_b[500:]."""
        merger = _load_module("chunk_merger")

        chunk_a = "PREFIX_A" + "X" * 500  # 508 chars total
        chunk_b = "Y" * 500 + "SUFFIX_B"  # 508 chars total

        mock_api = MagicMock()
        mock_api.generate_text.return_value = "MERGED_BOUNDARY"

        result = merger.merge_adjacent_chunks(chunk_a, chunk_b, mock_api)

        assert result.startswith("PREFIX_A"), "Result should start with prefix of chunk A"
        assert "MERGED_BOUNDARY" in result, "Result should contain merged boundary"
        assert result.endswith("SUFFIX_B"), "Result should end with suffix of chunk B"

    def test_merge_all_chunks_includes_part_markers(self):
        """Merged output should include part markers."""
        merger = _load_module("chunk_merger")

        # Test with 2 chunks - they get merged into one segment
        long_text_a = "A" * 1200 + " END_OF_PART_ONE"
        long_text_b = "START_PART_TWO " + "B" * 1200

        transcripts = [
            {"text": long_text_a},
            {"text": long_text_b},
        ]

        mock_api = MagicMock()
        mock_api.generate_text.return_value = "MERGED_BOUNDARY"

        result = merger.merge_all_chunks(transcripts, mock_api)

        # With new logic, 2 merged chunks show as [Parts 1-2]
        assert "[Parts 1-2]" in result, f"Should have Parts marker. Got: {result[:300]}"

    def test_merge_three_chunks_preserves_markers(self):
        """3-chunk merge should preserve markers (Gemini review fix)."""
        merger = _load_module("chunk_merger")

        transcripts = [
            {"text": "A" * 1200},
            {"text": "B" * 1200},
            {"text": "C" * 1200},
        ]

        mock_api = MagicMock()
        mock_api.generate_text.return_value = "MERGED"

        result = merger.merge_all_chunks(transcripts, mock_api)

        # All 3 chunks merged into one segment with [Parts 1-3] marker
        assert "[Parts 1-3]" in result, f"Should have Parts 1-3 marker. Got: {result[:300]}"

    def test_merge_all_handles_silence_chunks(self):
        """Silence chunks should be marked but not merged."""
        merger = _load_module("chunk_merger")

        transcripts = [
            {"text": "Normal speech here " + "X" * 600},
            {"text": "[SILENCE]"},
            {"text": "More speech " + "Y" * 600},
        ]

        mock_api = MagicMock()
        mock_api.generate_text.return_value = "MERGED"

        result = merger.merge_all_chunks(transcripts, mock_api)

        assert "[Part 2 - SILENCE]" in result, "Should mark silence chunks"
        assert "[Part 3]" in result, "Should have Part 3 marker after silence"

    def test_merge_handles_failed_chunks(self):
        """Failed transcription chunks should be marked, not merged."""
        merger = _load_module("chunk_merger")

        transcripts = [
            {"text": "Normal speech " + "X" * 600},
            {"text": "[TRANSCRIPTION FAILED]"},
            {"text": "More speech " + "Y" * 600},
        ]

        mock_api = MagicMock()
        mock_api.generate_text.return_value = "MERGED"

        result = merger.merge_all_chunks(transcripts, mock_api)

        assert "[Part 2 - TRANSCRIPTION FAILED]" in result, "Should mark failed chunks"


# =============================================================================
# CHUNKED TRANSCRIBER TESTS
# =============================================================================

class TestChunkedTranscriber:
    """transcribe_large runs the shared engine; these patch its seams (duration,
    splitting, the per-part provider request and reconstruction)."""

    @staticmethod
    def run(tmp_path, duration, results, reconstructed=None, chunks=2):
        import pytest
        engine = _load_module("engine")
        transcriber = _load_module("chunked_transcriber")
        audio_file = tmp_path / "long.m4a"
        audio_file.touch()
        made = []

        def split(path, chunk_sec, overlap_sec, output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out = []
            for i in range(chunks):
                chunk = Path(output_dir) / f"chunk_{i}.m4a"
                chunk.touch()
                made.append(chunk)
                out.append(MagicMock(path=str(chunk), start_time=i * 1800.0, end_time=(i + 1) * 1800.0 + 10))
            return out

        request = MagicMock(side_effect=results)
        merge = MagicMock(return_value=reconstructed or {"text": "Merged text", "segments": None,
                                                         "warnings": [], "boundaries": []})
        with patch.object(engine, "get_duration", return_value=duration), \
                patch.object(engine, "split_audio", side_effect=split) as splitter, \
                patch.object(engine, "_request", request), \
                patch.object(engine, "reconstruct", merge):
            try:
                result = transcriber.transcribe_large(str(audio_file))
                error = None
            except Exception as exc:  # returned for assertions below
                result, error = None, exc
        return dict(result=result, error=error, request=request, merge=merge, split=splitter, made=made)

    def test_small_file_uses_standard_transcription(self, tmp_path):
        """Files under 30 min go out as one request, unsplit."""
        run = self.run(tmp_path, 600.0, [{"text": "Short audio transcription", "success": True}])
        assert run["request"].call_count == 1
        run["split"].assert_not_called()
        assert run["result"]["text"] == "Short audio transcription"

    def test_large_file_splits_and_merges(self, tmp_path):
        """Files over 30 min are split, transcribed, and merged."""
        run = self.run(tmp_path, 3600.0, [{"text": "Chunk 1 text"}, {"text": "Chunk 2 text"}])
        assert run["request"].call_count == 2
        assert run["result"]["text"] == "Merged text"
        assert run["result"]["chunks"] == 2

    def test_returns_compatible_interface(self, tmp_path):
        """Return value must be compatible with TranscriptionResult dict."""
        run = self.run(tmp_path, 600.0, [{"text": "Test", "duration": 600.0, "processing_time": 30.0,
                                          "input_tokens": 1000, "output_tokens": 100,
                                          "model": "gemini-3-flash-preview", "success": True}])
        result = run["result"]
        for key in ("text", "duration", "success", "no_speech", "input_tokens", "output_tokens"):
            assert key in result
        assert isinstance(result["success"], bool)

    def test_cleans_up_temp_files(self, tmp_path):
        """Chunk files the engine created are deleted after transcription."""
        run = self.run(tmp_path, 3600.0, [{"text": "Test"}, {"text": "Test"}])
        assert run["made"] and not any(f.exists() for f in run["made"])

    def test_parse_error_chunk_does_not_kill_other_chunks(self, tmp_path):
        """A failed part does not stop the others: they are transcribed and merged, and the
        partial result travels on IncompleteTranscriptError instead of as a success."""
        from multimodal_transcription_core import IncompleteTranscriptError
        run = self.run(tmp_path, 3600.0, [
            {"text": "[PARSE_ERROR] finishReason=PROHIBITED_CONTENT, NO CONTENT returned"},
            {"text": "Good chunk 2 content", "success": True}])
        assert isinstance(run["error"], IncompleteTranscriptError)
        parts = run["merge"].call_args[0][0]
        assert parts[0].failed and not parts[1].failed
        assert parts[1].text == "Good chunk 2 content"
        assert run["error"].result["text"] == "Merged text" and run["error"].failed_parts == [1]

    def test_all_chunks_parse_error_raises(self, tmp_path):
        """Every part failing raises with the underlying reason, before any merge."""
        bad = {"text": "[PARSE_ERROR] finishReason=PROHIBITED_CONTENT, NO CONTENT returned"}
        run = self.run(tmp_path, 3600.0, [bad, bad])
        assert run["error"] is not None
        run["merge"].assert_not_called()
        assert "PARSE_ERROR" in str(run["error"]) or "PROHIBITED" in str(run["error"])

    def test_all_chunks_exception_failure_reraises_original(self, tmp_path):
        """If every part raises, the original exception propagates for classification."""
        run = self.run(tmp_path, 3600.0, [Exception("API error 401: invalid api_key"),
                                          Exception("API error 401: invalid api_key")])
        run["merge"].assert_not_called()
        assert "401" in str(run["error"]) and "api_key" in str(run["error"])

    def test_silence_plus_failure_raises(self, tmp_path):
        """Silence beside a failure is not a transcript: raise, never markers-only output."""
        run = self.run(tmp_path, 3600.0, [{"text": "[SILENCE]"},
                                          {"text": "[PARSE_ERROR] finishReason=PROHIBITED_CONTENT"}])
        assert run["error"] is not None and "PROHIBITED" in str(run["error"])
        run["merge"].assert_not_called()

    def test_empty_chunk_plus_failure_raises(self, tmp_path):
        """An empty/whitespace part is silence; with a failed part this must raise."""
        for empty_text in ("", "   ", "\n\n"):
            run = self.run(tmp_path, 3600.0, [{"text": empty_text},
                                              {"text": "[PARSE_ERROR] finishReason=PROHIBITED_CONTENT"}])
            assert run["error"] is not None and "PROHIBITED" in str(run["error"]), repr(empty_text)
            run["merge"].assert_not_called()

    def test_all_silence_is_valid_output(self, tmp_path):
        """All-silence is a valid, explicit no-speech result, not an error."""
        run = self.run(tmp_path, 3600.0, [{"text": "[SILENCE]"}, {"text": "[SILENCE]"}],
                       reconstructed={"text": "", "segments": None, "warnings": [], "boundaries": []})
        assert run["error"] is None
        assert run["result"]["no_speech"] is True and run["result"]["success"] is True
        assert run["result"]["text"] == "[SILENCE]"


# =============================================================================
# HALLUCINATION CLEANER TESTS
# =============================================================================

class TestHallucinationCleaner:
    """Tests for GeminiAPI hallucination cleaning methods."""

    def test_preserves_early_occurrence_of_repeated_word(self):
        """
        Bug fix: Cleaner should NOT truncate legitimate early occurrences.
        Example: "I said hello. hello hello hello" -> should keep "I said hello."
        """
        client = _load_module("gemini_client")
        api = client.GeminiAPI.__new__(client.GeminiAPI)

        # Text with legitimate "hello" early, then loop at end
        text = "I said hello to my friend. We talked for a while. " + "hello " * 50

        result = api._clean_hallucinations(text)

        assert "I said hello to my friend" in result, \
            f"Should preserve early legitimate occurrence. Got: {result[:100]}"
        assert result.count("hello") < 10, \
            f"Should remove the loop. Got {result.count('hello')} occurrences"

    def test_finds_contiguous_block_start(self):
        """_find_repetition_start should find where contiguous block begins."""
        client = _load_module("gemini_client")
        api = client.GeminiAPI.__new__(client.GeminiAPI)

        # Pattern "xyz" appears once early, then repeats at end
        text = "Some text with xyz in it. More content here. " + "xyz xyz xyz xyz xyz xyz xyz xyz"
        pattern = "xyz"

        start = api._find_repetition_start(text, pattern)

        # Should point to the start of the contiguous block, not the first occurrence
        assert start > 30, f"Should find block start near end, not early occurrence. Got: {start}"
        assert text[start:start+3] == "xyz", f"Position should point to pattern"

    def test_removes_severe_loops(self):
        """Should remove 50+ repetitions of a phrase."""
        client = _load_module("gemini_client")
        api = client.GeminiAPI.__new__(client.GeminiAPI)

        # Need substantial content before the loop (>30 chars)
        text = "This is valid content that should be preserved. The conversation was interesting. " + "repeat this " * 100

        result = api._clean_hallucinations(text)

        assert "This is valid content" in result
        assert len(result) < 500, f"Should truncate loop. Got {len(result)} chars"

    def test_marks_garbage_chunk(self):
        """If entire chunk is repetition, return [HALLUCINATION]."""
        client = _load_module("gemini_client")
        api = client.GeminiAPI.__new__(client.GeminiAPI)

        text = "garbage " * 100  # No real content

        result = api._clean_hallucinations(text)

        assert result == "[HALLUCINATION]", f"Should mark as hallucination. Got: {result[:50]}"

    def test_leaves_clean_text_unchanged(self):
        """Normal text without loops should pass through unchanged."""
        client = _load_module("gemini_client")
        api = client.GeminiAPI.__new__(client.GeminiAPI)

        text = "This is normal speech. It has no repetition patterns. Just regular conversation."

        result = api._clean_hallucinations(text)

        assert result == text, "Clean text should be unchanged"

    def test_detects_short_4char_pattern(self):
        """
        Bug fix: Cleaner must detect short patterns like 'Go. ' (4 chars).
        Previously only checked patterns >= 5 chars.
        """
        client = _load_module("gemini_client")
        api = client.GeminiAPI.__new__(client.GeminiAPI)

        # Real hallucination from chunk 9: "Oh shit. ... Go. Go. Go. Go." x800
        text = "Oh shit. Oh shit. Oh shit. " + "Go. " * 800

        result = api._clean_hallucinations(text)

        # Entire text is repetition, should be marked as hallucination
        assert result == "[HALLUCINATION]", f"Should detect short pattern. Got: {result[:100]}"


# =============================================================================
