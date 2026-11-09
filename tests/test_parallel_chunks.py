"""parallel_chunks: the parts of one long recording sent at the same time.

Each part is a different piece of audio sent once (not votes). Requests are faked;
BLIND SPOT: no real provider, so a provider's own concurrency limits are not exercised.
"""
import threading
import time
from types import SimpleNamespace

import pytest

from multimodel_transcription_kit import engine
from multimodel_transcription_kit.engine import ModelSpec
from multimodel_transcription_kit.rate_limiter import QuotaExhausted
from multimodel_transcription_kit.toml_config import ConfigError, convert_ini, load

TEXTS = ["one two three four five six", "three four five six seven eight", "five six seven eight nine ten"]
FULL = "one two three four five six seven eight nine ten"


def long_recording(tmp_path, monkeypatch, parts=3):
    """A recording that splits into `parts` chunks named part0.wav, part1.wav, ..."""
    audio = tmp_path / "long.wav"
    audio.touch()
    chunks = [SimpleNamespace(path=str(tmp_path / f"part{i}.wav"), start_time=i * 500.0,
                              end_time=i * 500.0 + 510) for i in range(parts)]
    monkeypatch.setattr(engine, "get_duration", lambda path: 500.0 * parts + 10)
    monkeypatch.setattr(engine, "split_audio", lambda *a, **k: chunks)
    monkeypatch.setattr(engine, "assess_audio", lambda path: {"problems": []})
    return audio


def spec(**extra):
    return ModelSpec(name="g", kind="gemini", model="gemini-3.6-flash", chunk_seconds=600, **extra)


def index(path):
    return int(str(path).rsplit("part", 1)[1].split(".")[0])


def test_parts_run_at_the_same_time_and_come_back_in_order(tmp_path, monkeypatch):
    audio = long_recording(tmp_path, monkeypatch)
    together = threading.Barrier(3, timeout=5)   # passes only if all three parts are in flight at once

    def request(spec, path, settings, clients):
        together.wait()
        time.sleep(0.05 * (3 - index(path)))      # the last part finishes first
        return {"text": TEXTS[index(path)], "no_speech": False}
    monkeypatch.setattr(engine, "_request", request)
    result = engine.transcribe_file(audio, spec(parallel_chunks=3), timestamps="none",
                                    speakers=False, sounds=False)
    assert result["text"] == FULL
    assert result["settings"]["parallel_chunks"] == 3


def test_parts_run_one_at_a_time_by_default(tmp_path, monkeypatch):
    audio = long_recording(tmp_path, monkeypatch)
    running, most, lock = [0], [0], threading.Lock()

    def request(spec, path, settings, clients):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        time.sleep(0.02)
        with lock:
            running[0] -= 1
        return {"text": TEXTS[index(path)], "no_speech": False}
    monkeypatch.setattr(engine, "_request", request)
    result = engine.transcribe_file(audio, spec(), timestamps="none", speakers=False, sounds=False)
    assert result["text"] == FULL and most[0] == 1
    assert spec().parallel_chunks == 1


def test_a_quota_error_stops_the_parts_not_yet_sent(tmp_path, monkeypatch):
    audio = long_recording(tmp_path, monkeypatch, parts=4)
    sent = []

    def request(spec, path, settings, clients):
        sent.append(index(path))
        if index(path) == 0:
            raise QuotaExhausted("g", time.time() + 60, "day used up")
        time.sleep(0.3)
        return {"text": "x", "no_speech": False}
    monkeypatch.setattr(engine, "_request", request)
    with pytest.raises(QuotaExhausted):
        engine.transcribe_file(audio, spec(parallel_chunks=2), timestamps="none", speakers=False, sounds=False)
    assert 0 in sent and set(sent) <= {0, 1}     # part 2 may already be in flight; 3 and 4 never go


@pytest.mark.parametrize("parallel", [1, 3])
def test_one_failed_part_keeps_the_others_the_same_way_in_both_modes(tmp_path, monkeypatch, parallel):
    audio = long_recording(tmp_path, monkeypatch)

    def request(spec, path, settings, clients):
        if index(path) == 1:
            raise engine.TranscriptionError("API error 400")
        return {"text": TEXTS[index(path)], "no_speech": False}
    monkeypatch.setattr(engine, "_request", request)
    with pytest.raises(engine.IncompleteTranscriptError) as caught:
        engine.transcribe_file(audio, spec(parallel_chunks=parallel), timestamps="none",
                               speakers=False, sounds=False)
    assert caught.value.failed_parts == [2]
    assert "one two" in caught.value.result["text"] and "nine ten" in caught.value.result["text"]


@pytest.mark.parametrize("value", ["0", "-2", "two"])
def test_parallel_chunks_must_be_a_positive_whole_number(value):
    with pytest.raises(ValueError, match="parallel_chunks"):
        ModelSpec.from_options("g", {"kind": "gemini", "parallel_chunks": value})


def test_model_spec_reads_parallel_chunks():
    assert ModelSpec.from_options("g", {"kind": "gemini", "parallel_chunks": "4"}).parallel_chunks == 4
    assert ModelSpec.from_options("g", {"kind": "gemini"}).parallel_chunks == 1


TOML = """
[[aistudio]]
name = "proxy"
key_env = "PROXY_KEY"
base_url = "http://127.0.0.1:9"
parallel_chunks = 4

[[aistudio]]
name = "free"
key_env = "FREE_KEY"

[[model]]
name = "m"
providers = ["proxy", "free"]
"""


def test_a_providers_parallel_chunks_reaches_every_route_through_it(tmp_path):
    path = tmp_path / "t.toml"
    path.write_text(TOML, encoding="utf-8")
    parser = load(path)
    proxy = ModelSpec.from_options("m@proxy", dict(parser["model.m@proxy"]))
    free = ModelSpec.from_options("m@free", dict(parser["model.m@free"]))
    assert (proxy.parallel_chunks, free.parallel_chunks) == (4, 1)


@pytest.mark.parametrize("value", ["0", "true", '"4"', "1.5"])
def test_a_bad_parallel_chunks_is_named(tmp_path, value):
    path = tmp_path / "t.toml"
    path.write_text(TOML.replace("parallel_chunks = 4", f"parallel_chunks = {value}"), encoding="utf-8")
    with pytest.raises(ConfigError, match="parallel_chunks"):
        load(path)


def test_ini_parallel_chunks_converts_to_the_provider():
    import configparser
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string("[models]\nranking = g\n[model.g]\nkind = gemini\nparallel_chunks = 3\n")
    assert "parallel_chunks = 3" in convert_ini(parser).split("[[model]]")[0]
