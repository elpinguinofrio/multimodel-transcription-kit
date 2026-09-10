"""Unit tests use synthetic credentials and cannot access external services."""
import socket
import pytest


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    monkeypatch.delenv("TRANSCRIPTION_ENV_FILE", raising=False)

    def deny_external(self, address):
        if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1"):
            raise AssertionError("External network access forbidden in offline tests")
        return original_connect(self, address)

    original_connect = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", deny_external)
