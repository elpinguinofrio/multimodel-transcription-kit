"""
Tests for Transcriber slice.

Uses direct file loading and patch.object to avoid package import issues.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib.util


def _load_transcriber():
    from importlib import import_module
    return import_module("multimodal_transcription_core.transcriber")


class TestTranscriber:
    """Test suite for audio transcription functionality."""

    def test_transcribe_returns_dict_with_text(self, tmp_path):
        """Returns dict with non-empty text from audio file."""
        audio_file = tmp_path / "test.m4a"
        audio_file.touch()

        transcriber = _load_transcriber()

        mock_api = MagicMock()
        mock_api.speech_to_text.return_value = {
            "text": "Hello world transcription",
            "model": "gemini-2.5-flash",
            "duration": 5.0,
            "processing_time": 2.0,
            "input_tokens": 100,
            "output_tokens": 50,
            "success": True
        }

        with patch.object(transcriber, "GeminiAPI", return_value=mock_api):
            result = transcriber.transcribe(audio_file)

        assert isinstance(result, dict)
        assert len(result["text"]) > 0
        assert result["text"] == "Hello world transcription"

    def test_transcribe_raises_on_invalid_file(self, tmp_path):
        """Raises error for non-existent file."""
        import pytest

        transcriber = _load_transcriber()
        non_existent = tmp_path / "does_not_exist.m4a"

        with pytest.raises(FileNotFoundError):
            transcriber.transcribe(non_existent)

    def test_transcribe_raises_on_non_audio_file(self, tmp_path):
        """Raises error for non-audio file extension."""
        import pytest

        transcriber = _load_transcriber()
        text_file = tmp_path / "notes.txt"
        text_file.write_text("not audio")

        with pytest.raises(ValueError, match="Unsupported"):
            transcriber.transcribe(text_file)

    def test_transcribe_uses_gemini_api(self, tmp_path):
        """Verifies GeminiAPI is called with correct parameters."""
        audio_file = tmp_path / "test.m4a"
        audio_file.touch()

        transcriber = _load_transcriber()

        mock_api = MagicMock()
        mock_api.speech_to_text.return_value = {"text": "Test"}

        with patch.object(transcriber, "GeminiAPI", return_value=mock_api):
            transcriber.transcribe(audio_file)

            mock_api.speech_to_text.assert_called_once()
            call_args = mock_api.speech_to_text.call_args
            assert str(audio_file) in str(call_args)

    def test_transcribe_handles_api_failure(self, tmp_path):
        """Raises RuntimeError if the Gemini API call fails."""
        import pytest

        audio_file = tmp_path / "test.m4a"
        audio_file.touch()

        transcriber = _load_transcriber()

        mock_api = MagicMock()
        mock_api.speech_to_text.side_effect = Exception("API error")

        with patch.object(transcriber, "GeminiAPI", return_value=mock_api):
            with pytest.raises(RuntimeError, match="Transcription failed"):
                transcriber.transcribe(audio_file)

    def test_transcribe_returns_transcription_result_dict(self, tmp_path):
        """Returns TranscriptionResult dict with all required fields."""
        audio_file = tmp_path / "test.m4a"
        audio_file.touch()

        transcriber = _load_transcriber()

        mock_api = MagicMock()
        mock_api.speech_to_text.return_value = {
            "text": "Hello world",
            "model": "gemini-2.5-flash",
            "duration": 5.0,
            "processing_time": 2.1,
            "input_tokens": 100,
            "output_tokens": 50,
            "success": True
        }

        with patch.object(transcriber, "GeminiAPI", return_value=mock_api):
            result = transcriber.transcribe(audio_file)

        # Result should be a dict, not a string
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result["text"] == "Hello world"
        assert result["model"] == "gemini-2.5-flash"
        assert "duration" in result
        assert "processing_time" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert result["success"] is True
