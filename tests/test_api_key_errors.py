"""A missing key error names the actual cause and the fix, never the key."""
import pytest

from multimodal_transcription_core import GeminiAPI, GeminiAPIError, GroqAPI, GroqAPIError

CLIENTS = [(GeminiAPI, GeminiAPIError, "GEMINI_API_KEY"), (GroqAPI, GroqAPIError, "GROQ_API_KEY")]


@pytest.mark.parametrize("client,error,var", CLIENTS)
def test_unset_key_without_env_file_says_to_name_one(monkeypatch, client, error, var):
    monkeypatch.delenv(var)
    with pytest.raises(error, match=f"Set {var}") as exc:
        client()
    assert "no TRANSCRIPTION_ENV_FILE" in str(exc.value)
    assert "TRANSCRIPTION_ENV_FILE=~/.config/transcription/.env" in str(exc.value)


@pytest.mark.parametrize("client,error,var", CLIENTS)
def test_blank_key_hiding_env_file_key_says_to_unset_it(tmp_path, monkeypatch, client, error, var):
    env = tmp_path / "keys.env"
    env.write_text(f"{var}=file-secret\n")
    monkeypatch.setenv(var, "")
    monkeypatch.setenv("TRANSCRIPTION_ENV_FILE", str(env))
    with pytest.raises(error) as exc:
        client()
    message = str(exc.value)
    assert f"{var} is set but empty" in message
    assert f"unset {var}" in message
    assert "file-secret" not in message


@pytest.mark.parametrize("client,error,var", CLIENTS)
def test_env_file_without_key_names_the_file(tmp_path, monkeypatch, client, error, var):
    env = tmp_path / "keys.env"
    env.write_text("OTHER=1\n")
    monkeypatch.delenv(var)
    monkeypatch.setenv("TRANSCRIPTION_ENV_FILE", str(env))
    with pytest.raises(error) as exc:
        client()
    assert f"{env} has no {var}" in str(exc.value)
