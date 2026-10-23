import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from multimodel_transcription_kit import GeminiAPI
from multimodel_transcription_kit.audio_chunker import AudioChunk, cleanup_chunks, split_audio
from multimodel_transcription_kit.config import load_env_file


def test_public_package_and_module_cli_use_the_kit_name():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    imported = subprocess.run(
        [sys.executable, "-c", "import json; from multimodel_transcription_kit.receipts import client_info; "
         "print(json.dumps(client_info()))"], env=env, capture_output=True, text=True, timeout=20)
    assert imported.returncode == 0, imported.stderr
    assert json.loads(imported.stdout)["package"] == "multimodel-transcription-kit"
    help_result = subprocess.run([sys.executable, "-m", "multimodel_transcription_kit", "--help"],
                                 env=env, capture_output=True, text=True, timeout=20)
    assert help_result.returncode == 0, help_result.stderr
    assert "multimodel-transcription-kit" in help_result.stdout


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
    with patch("multimodel_transcription_kit.audio_chunker.get_duration", return_value=120):
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
