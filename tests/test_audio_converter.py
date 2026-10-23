"""
Tests for AudioConverter metadata extraction.
"""

import importlib.util
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess


def _load_audio_converter():
    from importlib import import_module
    return import_module("multimodel_transcription_kit.audio_converter")


class TestGetMetadata:
    """Test suite for get_metadata function."""

    def test_get_metadata_returns_duration_and_creation_date(self):
        """get_metadata returns dict with duration and creation_date."""
        audio_converter = _load_audio_converter()

        # Mock ffprobe output with both duration and creation_time
        mock_output = """[FORMAT]
duration=125.340000
TAG:creation_time=2025-05-14T09:32:00.000000Z
[/FORMAT]"""

        with patch.object(subprocess, 'run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                returncode=0
            )

            result = audio_converter.AudioConverter.get_metadata("/fake/audio.m4a")

            assert result is not None
            assert "duration" in result
            assert "creation_date" in result
            assert result["duration"] == 125.34
            assert isinstance(result["creation_date"], datetime)

    def test_get_metadata_returns_none_creation_date_if_missing(self):
        """get_metadata falls back to file mtime if no creation_time tag."""
        audio_converter = _load_audio_converter()

        # Mock ffprobe output with only duration (no creation_time)
        mock_output = """[FORMAT]
duration=60.0
[/FORMAT]"""

        with patch.object(subprocess, 'run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                returncode=0
            )
            # Mock os.path.getmtime for fallback
            with patch('os.path.getmtime', return_value=1715680320.0):  # 2024-05-14 09:32:00 UTC
                result = audio_converter.AudioConverter.get_metadata("/fake/audio.wav")

                assert result is not None
                assert result["duration"] == 60.0
                assert result["creation_date"] is not None  # Should fallback to mtime

    def test_get_metadata_handles_ffprobe_failure(self):
        """get_metadata returns None if ffprobe fails."""
        audio_converter = _load_audio_converter()

        with patch.object(subprocess, 'run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")

            result = audio_converter.AudioConverter.get_metadata("/fake/audio.m4a")

            assert result is None
