import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from multimodal_transcription_core import GeminiAPI
from multimodal_transcription_core.audio_chunker import AudioChunk, cleanup_chunks, split_audio
from multimodal_transcription_core.config import load_env_file


def test_core_does_not_discover_private_parent_config(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=unrequested-secret\n")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(Exception, match="Set GEMINI_API_KEY"):
        GeminiAPI()


def test_explicit_env_file_and_shell_precedence(tmp_path, monkeypatch):
    env = tmp_path / "settings.env"
    env.write_text("GEMINI_API_KEY=file-key\nGEMINI_BASE_URL=http://example.invalid\n")
    monkeypatch.setenv("TRANSCRIPTION_ENV_FILE", str(env))
    api = GeminiAPI()
    assert api.api_key == "unit-test-key"
    assert api.base_url == "http://example.invalid"


def test_missing_explicit_env_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_ENV_FILE", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        load_env_file()


def test_cleanup_preserves_short_original(tmp_path):
    source = tmp_path / "original.wav"
    source.write_bytes(b"immutable-source")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    with patch("multimodal_transcription_core.audio_chunker.get_duration", return_value=120):
        chunks = split_audio(str(source))
    cleanup_chunks(chunks)
    assert source.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_cleanup_removes_only_owned_temporary_chunks(tmp_path):
    generated = tmp_path / "generated.wav"
    generated.touch()
    cleanup_chunks([AudioChunk(str(generated), 0, 1, 0, temporary=True)])
    assert not generated.exists()


@pytest.mark.parametrize("chunk_sec,overlap_sec", [(0, 10), (-1, 10), (30, -1)])
def test_invalid_chunk_intervals_fail_before_processing(chunk_sec, overlap_sec):
    with pytest.raises(ValueError):
        split_audio("unused.wav", chunk_sec=chunk_sec, overlap_sec=overlap_sec)
