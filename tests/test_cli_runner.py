"""Every cli-runner flag from PRODUCT.md, exercised against a stubbed provider.

BLIND SPOT: providers are stubbed here. Reachability probing and real transcript
content are covered by the integration tests, not by this file.
"""
import textwrap

import pytest

from multimodal_transcription_core import runner


CONFIG = """
[secrets]
env_file = {env_file}

[defaults]
language = ru

[models]
ranking = fast-cloud, slow-cloud, local-box

[model.fast-cloud]
kind = gemini

[model.slow-cloud]
kind = groq

[model.local-box]
kind = http
endpoint = http://127.0.0.1:9/v1/audio/transcriptions
"""


@pytest.fixture
def config_file(tmp_path):
    env = tmp_path / "keys.env"
    env.write_text("GEMINI_API_KEY=stub\n")
    path = tmp_path / "transcription.ini"
    path.write_text(textwrap.dedent(CONFIG).format(env_file=env))
    return path


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "my recording.m4a"
    path.write_bytes(b"stub-audio")
    return path


@pytest.fixture
def stub_provider(monkeypatch):
    """Record which model was asked for; return a fixed transcript."""
    calls = []

    def fake_transcribe(model_name, options, audio_path, language, limiter=None, **kwargs):
        calls.append({"model": model_name, "audio": audio_path, "language": language,
                      "limiter": limiter, **kwargs})
        return {"text": "стенограмма", "warnings": []}

    monkeypatch.setattr(runner, "transcribe_with_model", fake_transcribe)
    return calls


def reachable(monkeypatch, *names):
    monkeypatch.setattr(runner, "is_reachable", lambda name, options: name in names)


def test_no_destination_prints_transcript_to_stdout(capsys, config_file, audio, stub_provider, monkeypatch):
    reachable(monkeypatch, "fast-cloud")
    assert runner.main([str(audio), "--config", str(config_file)]) == 0
    assert capsys.readouterr().out == "стенограмма\n"


def test_destination_writes_named_markdown_with_source_header(config_file, audio, stub_provider, monkeypatch, tmp_path):
    reachable(monkeypatch, "fast-cloud")
    dest = tmp_path / "out"
    dest.mkdir()
    assert runner.main([str(audio), "-d", str(dest), "--config", str(config_file)]) == 0
    written = dest / "my recording.fast-cloud.md"
    assert written.is_file()
    assert written.read_text(encoding="utf-8") == f"src: {audio}\n\nстенограмма\n"


def test_repeat_runs_are_numbered_and_never_overwrite(config_file, audio, stub_provider, monkeypatch, tmp_path):
    reachable(monkeypatch, "fast-cloud")
    dest = tmp_path / "out"
    dest.mkdir()
    argv = [str(audio), "-d", str(dest), "--config", str(config_file)]
    for _ in range(3):
        assert runner.main(argv) == 0
    assert sorted(p.name for p in dest.iterdir()) == [
        "my recording.fast-cloud.(2).md",
        "my recording.fast-cloud.(3).md",
        "my recording.fast-cloud.md",
    ]


def test_model_flag_overrides_the_ranking(config_file, audio, stub_provider, monkeypatch):
    reachable(monkeypatch, "fast-cloud", "slow-cloud")
    assert runner.main([str(audio), "-m", "slow-cloud", "--config", str(config_file)]) == 0
    assert stub_provider[0]["model"] == "slow-cloud"


def test_default_model_is_the_first_reachable_in_ranking(config_file, audio, stub_provider, monkeypatch):
    reachable(monkeypatch, "slow-cloud", "local-box")
    assert runner.main([str(audio), "--config", str(config_file)]) == 0
    assert stub_provider[0]["model"] == "slow-cloud"


def test_language_flag_reaches_the_provider(config_file, audio, stub_provider, monkeypatch):
    reachable(monkeypatch, "fast-cloud")
    runner.main([str(audio), "--language", "en", "--config", str(config_file)])
    assert stub_provider[0]["language"] == "en"


def test_language_defaults_to_the_config_value(config_file, audio, stub_provider, monkeypatch):
    reachable(monkeypatch, "fast-cloud")
    runner.main([str(audio), "--config", str(config_file)])
    assert stub_provider[0]["language"] == "ru"


def test_unknown_model_is_rejected_before_any_provider_call(config_file, audio, stub_provider, monkeypatch, capsys):
    reachable(monkeypatch, "fast-cloud")
    with pytest.raises(SystemExit):
        runner.main([str(audio), "-m", "no-such-model", "--config", str(config_file)])
    assert stub_provider == []


def test_unreachable_requested_model_fails_loudly(config_file, audio, stub_provider, monkeypatch):
    reachable(monkeypatch)
    with pytest.raises(SystemExit):
        runner.main([str(audio), "-m", "fast-cloud", "--config", str(config_file)])
    assert stub_provider == []


def test_no_reachable_model_fails_loudly(config_file, audio, stub_provider, monkeypatch):
    reachable(monkeypatch)
    with pytest.raises(SystemExit):
        runner.main([str(audio), "--config", str(config_file)])
    assert stub_provider == []


def test_missing_audio_fails_before_config_work(config_file, tmp_path, stub_provider, monkeypatch):
    reachable(monkeypatch, "fast-cloud")
    with pytest.raises(SystemExit):
        runner.main([str(tmp_path / "absent.m4a"), "--config", str(config_file)])
    assert stub_provider == []


def test_missing_destination_directory_fails_loudly(config_file, audio, stub_provider, monkeypatch, tmp_path):
    reachable(monkeypatch, "fast-cloud")
    with pytest.raises(SystemExit):
        runner.main([str(audio), "-d", str(tmp_path / "absent"), "--config", str(config_file)])


def test_list_models_reports_reachability_without_transcribing(capsys, config_file, stub_provider, monkeypatch):
    reachable(monkeypatch, "slow-cloud")
    assert runner.main(["--list-models", "--config", str(config_file)]) == 0
    out = capsys.readouterr().out
    assert "fast-cloud" in out and "slow-cloud" in out and "local-box" in out
    assert stub_provider == []


def test_library_import_does_not_read_the_root_config(monkeypatch, tmp_path):
    """The library stays pure: only the CLI layer touches transcription.ini."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "transcription.ini").write_text("[models]\nranking = poisoned\n")
    import importlib

    import multimodal_transcription_core

    importlib.reload(multimodal_transcription_core)
    assert not hasattr(multimodal_transcription_core, "ranking")


LIMITED = CONFIG + """
[limits]
state_file = {state}
"""


@pytest.fixture
def limited_config(tmp_path):
    env = tmp_path / "keys.env"
    env.write_text("GEMINI_API_KEY=stub\n")
    path = tmp_path / "limited.ini"
    text = textwrap.dedent(LIMITED).format(env_file=env, state=tmp_path / "state.sqlite")
    text = text.replace("[model.fast-cloud]\nkind = gemini",
                        "[model.fast-cloud]\nkind = gemini\nrpm = 15\nrpd = 500")
    path.write_text(text)
    return path


def test_runner_passes_a_limiter_built_from_config(limited_config, audio, stub_provider, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert runner.main([str(audio), "--config", str(limited_config)]) == 0
    limiter = stub_provider[0]["limiter"]
    assert limiter.limits["fast-cloud"].rpd == 500


def test_list_models_shows_daily_usage(capsys, limited_config, stub_provider, monkeypatch):
    runner.main(["--list-models", "--config", str(limited_config)])
    out = capsys.readouterr().out
    assert "today 0/500" in out


def test_missing_config_points_at_the_committed_example(tmp_path, audio, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        runner.main([str(audio)])
    assert "transcription.example.ini" in capsys.readouterr().err


def test_committed_example_config_parses(capsys):
    from pathlib import Path
    from multimodal_transcription_core.rate_limiter import RateLimiter
    example = Path(__file__).resolve().parents[1] / "transcription.example.ini"
    limiter = RateLimiter.from_parser(runner.load_config(str(example)), state_path=None)
    assert limiter.limits["gemini-3.5-flash-lite"].rpd == 500


def _capture_options(monkeypatch):
    seen = []

    def fake_transcribe(model_name, options, audio_path, language, limiter=None, config=None, **kwargs):
        seen.append({"prompt": runner.model_spec(config, model_name, options).prompt})
        return {"text": "стенограмма", "warnings": []}

    monkeypatch.setattr(runner, "transcribe_with_model", fake_transcribe)
    return seen


def test_defaults_prompt_reaches_the_chosen_model(tmp_path, audio, monkeypatch):
    path = tmp_path / "transcription.ini"
    path.write_text(textwrap.dedent(CONFIG).format(env_file=tmp_path / "none.env")
                    .replace("language = ru", "language = ru\nprompt = Write a verbatim transcript."))
    seen = _capture_options(monkeypatch)
    reachable(monkeypatch, "fast-cloud")
    assert runner.main([str(audio), "--config", str(path)]) == 0
    assert seen[0]["prompt"] == "Write a verbatim transcript."


def test_model_prompt_overrides_defaults_prompt(tmp_path, audio, monkeypatch):
    path = tmp_path / "transcription.ini"
    path.write_text(textwrap.dedent(CONFIG).format(env_file=tmp_path / "none.env")
                    .replace("language = ru", "language = ru\nprompt = default text")
                    .replace("[model.fast-cloud]\nkind = gemini", "[model.fast-cloud]\nkind = gemini\nprompt = model text"))
    seen = _capture_options(monkeypatch)
    reachable(monkeypatch, "fast-cloud")
    assert runner.main([str(audio), "--config", str(path)]) == 0
    assert seen[0]["prompt"] == "model text"


def test_gemini_dispatch_passes_the_prompt(monkeypatch, audio):
    from multimodal_transcription_core import gemini_client
    calls = []

    class FakeGemini:
        def __init__(self, **kwargs):
            pass

        def speech_to_text(self, path, model=None, language=None, prompt=None, **kwargs):
            calls.append(prompt)
            return {"text": "ok", "no_speech": False}

    monkeypatch.setattr(gemini_client, "GeminiAPI", FakeGemini)
    result = runner.transcribe_with_model("m", {"kind": "gemini", "prompt": "VERBATIM"}, audio, "ru")
    assert result["text"] == "ok"
    assert calls == ["VERBATIM"]


def test_defaults_prompt_is_not_sent_to_whisper_models(tmp_path):
    import configparser
    config = configparser.ConfigParser()
    config.read_string("[defaults]\nprompt = Gemini instructions\n")
    assert runner.model_spec(config, "w", {"kind": "groq"}).prompt is None
    assert runner.model_spec(config, "g", {"kind": "gemini"}).prompt == "Gemini instructions"


def test_example_config_ships_a_verbatim_default_prompt():
    import configparser
    from pathlib import Path
    parser = configparser.ConfigParser()
    parser.read(Path(__file__).resolve().parents[1] / "transcription.example.ini", encoding="utf-8")
    assert "verbatim" in parser.get("defaults", "prompt").lower()
