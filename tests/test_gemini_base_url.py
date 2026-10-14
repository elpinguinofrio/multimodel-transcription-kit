"""
Tests for GEMINI_BASE_URL override (alternate Gemini-compatible endpoint).

The Gemini client must be routable to an alternate endpoint via env var,
with no other hardcoded endpoints. Default behavior must stay unchanged.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib.util


def _load_module(name: str):
    from importlib import import_module
    return import_module(f"multimodal_transcription_core.{name}")


GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ALT_BASE_URL = "http://127.0.0.1:8300"


class TestBaseUrlOverride:
    """GeminiAPI must resolve its endpoint from GEMINI_BASE_URL."""

    def test_default_url_unchanged_without_env(self, monkeypatch):
        monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        gemini_client = _load_module("gemini_client")
        # find_dotenv resolves from the module dir, so the repo's real .env
        # would leak in regardless of cwd — stub the loader for isolation.
        monkeypatch.setattr(gemini_client.GeminiAPI, "_load_env_files",
                            staticmethod(lambda: None))
        api = gemini_client.GeminiAPI()
        assert api.API_URL == GOOGLE_URL

    def test_env_override_swaps_host_keeps_path(self, monkeypatch):
        monkeypatch.setenv("GEMINI_BASE_URL", ALT_BASE_URL + "/")  # trailing slash tolerated
        monkeypatch.setenv("GEMINI_API_KEY", "dummy")
        gemini_client = _load_module("gemini_client")
        api = gemini_client.GeminiAPI()
        assert api.API_URL == ALT_BASE_URL + "/v1beta/models/{model}:generateContent"

    def test_base_url_from_dotenv_when_key_already_in_shell(self, monkeypatch):
        """A shell-exported GEMINI_API_KEY must not short-circuit .env loading,
        or GEMINI_BASE_URL defined only in .env would be silently ignored."""
        import os
        monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "shell-key")
        gemini_client = _load_module("gemini_client")

        def fake_env_load():  # stands in for a .env that defines only the base URL
            os.environ.setdefault("GEMINI_BASE_URL", ALT_BASE_URL)

        monkeypatch.setattr(gemini_client.GeminiAPI, "_load_env_files",
                            staticmethod(fake_env_load))
        try:
            api = gemini_client.GeminiAPI()
            assert api.API_URL == ALT_BASE_URL + "/v1beta/models/{model}:generateContent"
        finally:
            os.environ.pop("GEMINI_BASE_URL", None)

    def test_speech_to_text_posts_to_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEMINI_BASE_URL", ALT_BASE_URL)
        monkeypatch.setenv("GEMINI_API_KEY", "dummy")
        gemini_client = _load_module("gemini_client")
        api = gemini_client.GeminiAPI()

        audio = tmp_path / "t.mp3"
        audio.write_bytes(b"\x00" * 16)

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            res = MagicMock()
            res.status_code = 200
            res.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "hello"}]}}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            }
            return res

        with patch.object(gemini_client, "get_duration", return_value=1.0, create=True), \
             patch.object(gemini_client.requests, "post", side_effect=fake_post):
            api.speech_to_text(audio_input=str(audio),
                               model=gemini_client.GeminiModel.GEMINI_3_5_FLASH,
                               response_format="text")

        assert captured["url"].startswith(ALT_BASE_URL), captured["url"]
        assert "generativelanguage.googleapis.com" not in captured["url"]
        assert "key=dummy" in captured["url"]


class TestListModelsUsesBaseUrl:
    """list_available_models() must hit the configured base URL, not hardcode Google."""

    def test_list_models_uses_override(self, monkeypatch):
        monkeypatch.setenv("GEMINI_BASE_URL", ALT_BASE_URL)
        monkeypatch.setenv("GEMINI_API_KEY", "dummy")
        gemini_client = _load_module("gemini_client")
        api = gemini_client.GeminiAPI()

        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            res = MagicMock()
            res.status_code = 200
            res.json.return_value = {"models": []}
            return res

        with patch.object(gemini_client.requests, "get", side_effect=fake_get):
            api.list_available_models()

        assert captured["url"].startswith(ALT_BASE_URL), captured["url"]
        assert "generativelanguage.googleapis.com" not in captured["url"]


class TestMergeApiViaBaseUrl:
    """ChunkedMergeAPI.generate_text must follow the overridden base URL."""

    def test_generate_text_posts_to_override(self, monkeypatch):
        monkeypatch.setenv("GEMINI_BASE_URL", ALT_BASE_URL)
        monkeypatch.setenv("GEMINI_API_KEY", "dummy")
        gemini_client = _load_module("gemini_client")
        chunked = _load_module("chunked_transcriber")

        api = gemini_client.GeminiAPI()
        merge = chunked.ChunkedMergeAPI(api)

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            res = MagicMock()
            res.status_code = 200
            res.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "merged"}]}}]
            }
            return res

        with patch("requests.post", side_effect=fake_post):
            out = merge.generate_text("merge these")

        assert out == "merged"
        assert captured["url"].startswith(ALT_BASE_URL), captured["url"]
        assert "generativelanguage.googleapis.com" not in captured["url"]


class TestChunkedModelIsPriced:
    """Chunked path must use the supported replacement: gemini-3.6-flash."""

    def test_merge_model_is_3_6_flash(self):
        chunked = _load_module("chunked_transcriber")
        assert chunked.MERGE_MODEL == "gemini-3.6-flash"

    def test_transcribe_chunk_uses_3_6_flash(self, tmp_path):
        chunked = _load_module("chunked_transcriber")
        mock_api = MagicMock()
        mock_api.speech_to_text.return_value = {"text": "x", "success": True}
        chunked._transcribe_chunk(str(tmp_path / "c.mp3"), mock_api)
        model = mock_api.speech_to_text.call_args.kwargs["model"]
        assert model.value == "gemini-3.6-flash"
