"""Profiles: named model orders in one config, e.g. "quality" for content and "balanced" for bulk.

A caller picks one with --profile; without it the config's [defaults] profile applies,
and without profiles the [[model]] block order is the order. Providers are faked.
"""
import configparser

import pytest

from multimodel_transcription_kit import runner
from multimodel_transcription_kit.toml_config import ConfigError, convert_ini, load
from test_engine import tone

CONFIG = """
[defaults]
{default}

[[aistudio]]
name = "free"
key_env = "FREE_KEY"

[[aistudio]]
name = "paid"
key_env = "PAID_KEY"
costs_money = true

[[model]]
name = "lite"
providers = ["free", "paid"]

[[model]]
name = "flash"
providers = ["free", "paid"]

[[model]]
name = "whisper"
providers = ["free"]
word_times = true

[[profile]]
name = "quality"
models = ["flash", "lite"]

[[profile]]
name = "balanced"
models = ["lite", "flash", "whisper"]
"""


def write(tmp_path, text):
    path = tmp_path / "t.toml"
    path.write_text(text, encoding="utf-8")
    return path


def config(tmp_path, default='profile = "quality"'):
    return load(write(tmp_path, CONFIG.format(default=default)))


def test_each_profile_is_its_own_order_with_paid_routes_last(tmp_path):
    parser = config(tmp_path)
    assert runner.ranked_models(parser, "quality") == ["flash@free", "lite@free", "flash@paid", "lite@paid"]
    assert runner.ranked_models(parser, "balanced") == ["lite@free", "flash@free", "whisper@free",
                                                        "lite@paid", "flash@paid"]


def test_the_default_profile_applies_when_the_caller_names_none(tmp_path):
    assert runner.ranked_models(config(tmp_path)) == runner.ranked_models(config(tmp_path), "quality")
    assert runner.ranked_models(config(tmp_path, 'profile = "balanced"'))[0] == "lite@free"


def test_without_profiles_the_model_block_order_is_the_order(tmp_path):
    text = CONFIG.format(default="").split("[[profile]]")[0]
    parser = load(write(tmp_path, text))
    assert runner.ranked_models(parser) == ["lite@free", "flash@free", "whisper@free", "lite@paid", "flash@paid"]
    assert runner.profile_names(parser) == []


def test_word_times_keep_the_model_block_order_in_every_profile(tmp_path):
    assert runner.timing_models(config(tmp_path)) == ["whisper@free"]


@pytest.mark.parametrize("extra,default,message", [
    ('[[profile]]\nname = "x"\nmodels = ["nope"]\n', "", "no model named 'nope'"),
    ('[[profile]]\nname = "quality"\nmodels = ["lite"]\n', "", "two profiles are named 'quality'"),
    ('[[profile]]\nname = "x"\n', "", "list its models"),
    ('[[profile]]\nname = "x"\nmodels = ["lite"]\nspeed = 1\n', "", "unknown setting"),
    ("", 'profile = "fast"', "no profile named 'fast'"),
])
def test_profile_mistakes_are_named(tmp_path, extra, default, message):
    with pytest.raises(ConfigError, match=message.replace("[", r"\[")):
        load(write(tmp_path, CONFIG.format(default=default) + extra))


def test_profiles_need_a_default_or_the_block_order_applies(tmp_path):
    parser = config(tmp_path, default="")
    assert runner.ranked_models(parser) == ["lite@free", "flash@free", "whisper@free", "lite@paid", "flash@paid"]
    assert runner.profile_names(parser) == ["quality", "balanced"]


def _calls(tmp_path, monkeypatch, *flags, default='profile = "quality"'):
    monkeypatch.setenv("FREE_KEY", "k")
    monkeypatch.setenv("PAID_KEY", "k")
    calls = []

    def transcribe_with_model(name, options, audio_path, language, **kw):
        calls.append(name)
        return {"text": "x", "name": name, "warnings": []}
    monkeypatch.setattr(runner, "transcribe_with_model", transcribe_with_model)
    monkeypatch.setattr(runner, "is_reachable", lambda name, options: True)
    path = write(tmp_path, CONFIG.format(default=default))
    assert runner.main([str(tone(tmp_path / "a.wav", 1)), "--config", str(path), *flags]) == 0
    return calls


def test_cli_uses_the_default_profile(tmp_path, monkeypatch):
    assert _calls(tmp_path, monkeypatch) == ["flash@free"]


def test_cli_profile_flag_picks_another_order(tmp_path, monkeypatch):
    assert _calls(tmp_path, monkeypatch, "--profile", "balanced") == ["lite@free"]


def test_cli_unknown_profile_lists_the_known_ones(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _calls(tmp_path, monkeypatch, "--profile", "fast")
    assert "quality, balanced" in capsys.readouterr().err


def test_ini_profiles_list_model_sections():
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string("[defaults]\nprofile = quality\n[models]\nranking = a, b\n"
                       "[profiles]\nquality = b, a\nbalanced = a\n"
                       "[model.a]\nkind = gemini\n[model.b]\nkind = gemini\n")
    assert runner.ranked_models(parser) == ["b", "a"]
    assert runner.ranked_models(parser, "balanced") == ["a"]
    converted = convert_ini(parser)
    assert '[[profile]]\nname = "quality"\nmodels = ["b", "a"]' in converted


def test_example_toml_has_quality_and_balanced_profiles():
    from pathlib import Path
    parser = load(Path(__file__).resolve().parents[1] / "transcription.example.toml")
    assert runner.profile_names(parser) == ["quality", "balanced"]
    assert runner.ranked_models(parser, "quality")[0].startswith("gemini-3.8-flash@")
    assert runner.ranked_models(parser, "balanced")[0].startswith("gemini-3.5-flash-lite@")


def test_example_toml_sets_each_models_longest_part():
    """Every model states its longest part; flash-lite's is 10 minutes (seen live 2026-11-09:
    30-minute parts hit the output limit, and one 18-minute part stopped 14 minutes in)."""
    import tomllib
    from pathlib import Path
    data = tomllib.loads((Path(__file__).resolve().parents[1] / "transcription.example.toml").read_text())
    lengths = {model["name"]: model.get("chunk_seconds") for model in data["model"]}
    assert all(lengths.values()), [name for name, seconds in lengths.items() if not seconds]
    assert lengths["gemini-3.5-flash-lite"] == 600
