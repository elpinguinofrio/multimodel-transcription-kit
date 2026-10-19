"""Every transcribe_file setting a person chooses has a CLI flag (docs/PRODUCT.md: Simple CLI).

A flag overrides [defaults]; without it the config or the library default applies.
"""
import pytest

from multimodal_transcription_core import engine, runner

CONFIG = "[defaults]\noverlap_seconds = 12\n[models]\nranking = w\n[model.w]\nkind = groq\n"


def run(tmp_path, monkeypatch, *flags):
    audio = tmp_path / "a.wav"
    audio.touch()
    config = tmp_path / "t.ini"
    config.write_text(CONFIG)
    seen = {}
    monkeypatch.setattr(engine, "transcribe_file", lambda audio, spec, **kw: seen.update(kw) or {"text": "t"})
    assert runner.main([str(audio), "--config", str(config), *flags]) == 0
    return seen


@pytest.mark.parametrize("flags,key,value", [
    (["--prompt", "Write it verbatim."], "prompt", "Write it verbatim."),
    (["--temperature", "0.3"], "temperature", 0.3),
    (["--overlap-seconds", "5"], "overlap_seconds", 5.0),
    (["--max-retries", "2"], "max_retries", 2),
    (["--timeout", "90"], "timeout", 90.0),
])
def test_flag_reaches_the_library(tmp_path, monkeypatch, flags, key, value):
    assert run(tmp_path, monkeypatch, *flags)[key] == value


def test_without_flags_config_and_library_defaults_apply(tmp_path, monkeypatch):
    seen = run(tmp_path, monkeypatch)
    assert seen["overlap_seconds"] == 12.0
    assert not {"prompt", "temperature", "max_retries", "timeout"} & set(seen)
