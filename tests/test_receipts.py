"""Benchmark runs are reproducible from their receipts; accuracy needs a reviewed human reference.

Providers are the synthetic local gateway from test_engine. BLIND SPOT: no real provider.
"""
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import multimodal_transcription_core as core
from multimodal_transcription_core import bench, engine, runner
from test_engine import gateway, tone


def test_gemini_result_records_the_exact_request(monkeypatch):
    api = core.GeminiAPI(throttle_rate=0)
    monkeypatch.setattr(api, "_rate_limit", lambda: None)
    response = Mock(status_code=200)
    response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
    monkeypatch.setattr(api, "_post_with_retry", lambda *a, **kw: response)
    result = api.speech_to_text(b"x", prompt="VERBATIM", temperature=0.2)
    assert "VERBATIM" in result["request"]["prompt"]
    assert result["request"]["generationConfig"]["temperature"] == 0.2
    assert result["request"]["model"] == "gemini-3-flash-preview"


@pytest.mark.parametrize("kind", ["gemini", "groq", "http"])
def test_engine_keeps_raw_responses_and_client_version_when_asked(tmp_path, monkeypatch, kind):
    from test_engine import spec_for
    audio = tone(tmp_path / "a.wav", 2)
    with gateway(kind) as (url, _):
        monkeypatch.setenv("GEMINI_BASE_URL", url)
        monkeypatch.setattr(core.GroqAPI, "API_URL", f"{url}/openai/v1/audio/transcriptions")
        result = engine.transcribe_file(audio, spec_for(kind, url), keep_raw=True, limiter=core.RateLimiter({}))
    part = result["parts"][0]
    assert part["raw_response"] and part["request"]["model"] == spec_for(kind, url).model
    assert result["client"]["package"] == "multimodal-transcription-core" and result["client"]["version"]


CONFIG = """
[models]
ranking = local
[model.local]
kind = http
model = parakeet
endpoint = {url}/v1/audio/transcriptions
"""


def test_cli_receipt_sits_beside_the_transcript(tmp_path):
    audio = tone(tmp_path / "talk.wav", 2)
    out = tmp_path / "out"
    out.mkdir()
    config = tmp_path / "t.ini"
    with gateway("http") as (url, _):
        config.write_text(CONFIG.format(url=url))
        for _ in range(2):
            assert runner.main([str(audio), "--config", str(config), "-d", str(out), "--receipt"]) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["talk.local.(2).md", "talk.local.(2).receipt.json", "talk.local.md", "talk.local.receipt.json"]
    receipt = json.loads((out / "talk.local.receipt.json").read_text())
    assert receipt["audio"]["sha256"] == hashlib.sha256(audio.read_bytes()).hexdigest()
    assert receipt["spec"]["model"] == "parakeet" and receipt["parts"][0]["raw_response"]["text"] == "hello world"
    assert receipt["transcript_file"] == "talk.local.md"


def run_bench(tmp_path, url, *extra):
    config = tmp_path / "t.ini"
    config.write_text(CONFIG.format(url=url))
    audio = tmp_path / "sample.wav"
    if not audio.exists():
        tone(audio, 2)
    runs = tmp_path / "runs"
    return bench.main(["--config", str(config), "--audio", str(audio), "--runs-dir", str(runs),
                       "-m", "local", "--label", "p1", *extra]), runs


def test_bench_versions_runs_and_never_overwrites(tmp_path):
    with gateway("http") as (url, _):
        assert run_bench(tmp_path, url)[0] == 0
        first = (tmp_path / "runs" / "parakeet__p1__v001.txt").read_bytes()
        assert run_bench(tmp_path, url, "--repeat", "2")[0] == 0
    runs = tmp_path / "runs"
    assert sorted(p.name for p in runs.glob("*.txt")) == [
        "parakeet__p1__v001.txt", "parakeet__p1__v002.txt", "parakeet__p1__v003.txt"]
    assert (runs / "parakeet__p1__v001.txt").read_bytes() == first
    receipts = sorted(p.name for p in (runs / "_details" / "receipts").iterdir())
    assert receipts == ["parakeet__p1__v001.json", "parakeet__p1__v002.json", "parakeet__p1__v003.json"]


def test_failed_bench_runs_keep_a_receipt(tmp_path):
    with gateway("http", statuses=(403,)) as (url, _):
        code, runs = run_bench(tmp_path, url)
    assert code == 1
    receipt = json.loads((runs / "_details" / "receipts" / "parakeet__p1__v001.json").read_text())
    assert receipt["status"] == "failed" and "403" in receipt["error"]
    assert not list(runs.glob("*.txt"))


def test_a_receipt_replays_the_same_request(tmp_path):
    with gateway("http") as (url, seen):
        run_bench(tmp_path, url, "--language", "ru")
        receipt = tmp_path / "runs" / "_details" / "receipts" / "parakeet__p1__v001.json"
        assert bench.main(["--replay", str(receipt), "--runs-dir", str(tmp_path / "runs")]) == 0
    first, second = (s["body"] for s in seen)
    strip = lambda body: body.split(b"--benchmark-")[1:]          # boundaries differ per request
    assert [p.split(b"\r\n", 1)[1] for p in strip(first)] == [p.split(b"\r\n", 1)[1] for p in strip(second)]
    replay = json.loads((tmp_path / "runs" / "_details" / "receipts" / "parakeet__p1__v002.json").read_text())
    assert replay["replay_of"].endswith("parakeet__p1__v001.json")
    assert replay["options"]["language"] == "ru"


def test_replay_refuses_changed_audio(tmp_path):
    with gateway("http") as (url, _):
        run_bench(tmp_path, url)
        tone(tmp_path / "sample.wav", 3)
        receipt = tmp_path / "runs" / "_details" / "receipts" / "parakeet__p1__v001.json"
        with pytest.raises(SystemExit):
            bench.main(["--replay", str(receipt), "--runs-dir", str(tmp_path / "runs")])


# ---------------------------------------------------------------- scorer provenance

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_transcripts.py"


def scorer():
    spec = spec_from_file_location("score_transcripts", SCRIPT)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_with(tmp_path, record=None):
    reference = tmp_path / "reference.txt"
    reference.write_text("раз два три четыре", encoding="utf-8")
    hypothesis = tmp_path / "model.txt"
    hypothesis.write_text("раз два три", encoding="utf-8")
    if record is not None:
        record = {"reference": "reference.txt", "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                  **record}
        (tmp_path / "reference.local.json").write_text(json.dumps(record))
    return reference, hypothesis


REVIEWED = {"origin": "human-from-audio", "status": "reviewed", "reviewed_by": "owner", "reviewed_on": "2026-10-14"}


def test_scores_need_a_reviewed_human_reference(tmp_path):
    reference, hypothesis = reference_with(tmp_path)
    with pytest.raises(SystemExit):
        scorer().main(["--reference", str(reference), "--hypothesis", str(hypothesis),
                       "--output", str(tmp_path / "s.json")])
    assert not (tmp_path / "s.json").exists()


def test_unreviewed_scores_are_labelled_as_no_accuracy_claim(tmp_path):
    reference, hypothesis = reference_with(tmp_path)
    scorer().main(["--reference", str(reference), "--hypothesis", str(hypothesis),
                   "--output", str(tmp_path / "s.json"), "--allow-unreviewed"])
    report = json.loads((tmp_path / "s.json").read_text())
    assert report["accuracy_claim"] is False and "unreviewed" in report["status"]


def test_reviewed_human_reference_supports_accuracy_claims(tmp_path):
    reference, hypothesis = reference_with(tmp_path, REVIEWED)
    scorer().main(["--reference", str(reference), "--hypothesis", str(hypothesis), "--output", str(tmp_path / "s.json")])
    report = json.loads((tmp_path / "s.json").read_text())
    assert report["accuracy_claim"] is True and report["review"]["reviewed_by"] == "owner"
    assert report["results"][0]["deletions"] == 1


@pytest.mark.parametrize("record", [
    {**REVIEWED, "origin": "machine-seeded"}, {**REVIEWED, "status": "draft"}, {**REVIEWED, "sha256": "0" * 64},
])
def test_machine_seeded_draft_or_changed_references_are_refused(tmp_path, record):
    reference, hypothesis = reference_with(tmp_path, record)
    with pytest.raises(SystemExit):
        scorer().main(["--reference", str(reference), "--hypothesis", str(hypothesis),
                       "--output", str(tmp_path / "s.json")])


def test_bench_fuse_writes_a_fused_run_with_its_inputs(tmp_path, monkeypatch):
    from multimodal_transcription_core import fusion
    monkeypatch.setattr(fusion, "gemini_fuser", lambda model, **kw: lambda texts: "hello world")
    config = tmp_path / "t.ini"
    with gateway("http") as (url, _):
        config.write_text(CONFIG.format(url=url) + f"[model.other]\nkind = http\nmodel = other\n"
                          f"endpoint = {url}/v1/audio/transcriptions\n")
        audio = tone(tmp_path / "sample.wav", 2)
        assert bench.main(["--config", str(config), "--audio", str(audio), "--runs-dir", str(tmp_path / "runs"),
                           "-m", "local", "-m", "other", "--label", "p1", "--fuse"]) == 0
    runs = tmp_path / "runs"
    assert sorted(p.name for p in runs.glob("*.txt")) == [
        "fused__p1__v001.txt", "other__p1__v001.txt", "parakeet__p1__v001.txt"]
    receipt = json.loads((runs / "_details" / "receipts" / "fused__p1__v001.json").read_text())
    assert receipt["fusion"]["inputs"] == ["parakeet__p1__v001.txt", "other__p1__v001.txt"]
    assert receipt["fusion"]["model"] == "gemini-3.6-flash" and receipt["status"] == "ok"


def test_bench_applies_the_config_defaults_like_the_cli(tmp_path):
    from test_engine import PARTS, needs_ffmpeg  # noqa: F401  (ffmpeg checked below)
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("requires ffmpeg")
    with gateway("http", PARTS) as (url, seen):
        config = tmp_path / "t.ini"
        config.write_text("[defaults]\nlanguage = ru\noverlap_seconds = 2\n" + CONFIG.format(url=url)
                          + "chunk_seconds = 10\n")
        audio = tone(tmp_path / "long.wav", 25)
        assert bench.main(["--config", str(config), "--audio", str(audio), "--runs-dir", str(tmp_path / "runs"),
                           "-m", "local", "--label", "p1"]) == 0
    receipt = json.loads((tmp_path / "runs" / "_details" / "receipts" / "parakeet__p1__v001.json").read_text())
    assert receipt["options"]["overlap_seconds"] == 2.0 and receipt["options"]["language"] == "ru"
    assert receipt["result"]["chunks"] == 4
