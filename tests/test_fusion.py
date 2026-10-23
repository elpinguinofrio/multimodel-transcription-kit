"""Several models on one recording, combined; individual outputs always kept.

Owner decision (2026-10-14): record every failure; fuse when at least two models
succeed; one success is returned labelled "not fused"; none raises.
BLIND SPOT: the fusion model is stubbed; whether fusion beats the best single model
is measured only by the scorer against a reviewed human reference.
"""
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import multimodel_transcription_kit as core
from multimodel_transcription_kit import fusion, runner
from multimodel_transcription_kit.engine import ModelSpec

SPECS = [ModelSpec(name=n, kind="groq", model=f"whisper-{n}") for n in ("a", "b", "c")]


def transcriber(fail=()):
    def transcribe(audio, spec, **kwargs):
        if spec.name in fail:
            raise core.ProviderError(f"{spec.name} down")
        return {"text": f"[0:00:00] words from {spec.name}", "plain_text": f"words from {spec.name}",
                "name": spec.name, "model": spec.model, "provider": spec.kind, "warnings": []}
    return transcribe


def fuser(seen, output="words from fused"):
    def fuse(texts):
        seen.append(texts)
        return output
    return fuse


def test_successful_models_are_fused_and_kept():
    seen = []
    result = fusion.transcribe_fused("x.wav", SPECS[:2], transcribe=transcriber(), fuse_texts=fuser(seen))
    assert seen == [["words from a", "words from b"]]          # plain text, no timestamps
    assert result["text"] == "words from fused" and result["fusion"]["status"] == "fused"
    assert [r["name"] for r in result["individual"]] == ["a", "b"]


def test_survivors_are_fused_and_failures_recorded():
    seen = []
    result = fusion.transcribe_fused("x.wav", SPECS, transcribe=transcriber(fail={"b"}), fuse_texts=fuser(seen))
    assert seen == [["words from a", "words from c"]]
    assert result["fusion"]["inputs"] == ["a", "c"]
    assert result["fusion"]["failures"][0]["name"] == "b" and "down" in result["fusion"]["failures"][0]["error"]


def test_a_single_success_is_returned_unfused():
    seen = []
    result = fusion.transcribe_fused("x.wav", SPECS[:2], transcribe=transcriber(fail={"a"}), fuse_texts=fuser(seen))
    assert seen == [] and result["fusion"]["status"] == "not fused"
    assert result["text"] == "[0:00:00] words from b"


def test_no_success_raises_with_every_failure():
    with pytest.raises(core.NoModelAvailableError) as caught:
        fusion.transcribe_fused("x.wav", SPECS[:2], transcribe=transcriber(fail={"a", "b"}), fuse_texts=fuser([]))
    assert [a["name"] for a in caught.value.attempts] == ["a", "b"]


def test_unreachable_members_are_recorded_as_failures():
    result = fusion.transcribe_fused("x.wav", SPECS, transcribe=transcriber(), fuse_texts=fuser([]),
                                     probe=lambda spec: spec.name != "c")
    assert result["fusion"]["failures"] == [{"name": "c", "error": "not reachable"}]


@pytest.mark.parametrize("output", ["", "words", "words " * 50])
def test_fused_output_that_drops_or_invents_text_is_rejected(output):
    with pytest.raises(core.FusionError) as caught:
        fusion.transcribe_fused("x.wav", SPECS[:2], transcribe=transcriber(), fuse_texts=fuser([], output))
    assert [r["name"] for r in caught.value.individual] == ["a", "b"]


def test_fusion_prompt_contains_every_transcript():
    prompt = fusion.fusion_prompt(["one two", "uno dos"])
    assert "one two" in prompt and "uno dos" in prompt and "never add" in prompt.lower()


def test_truncated_generation_is_an_error_not_text(monkeypatch):
    api = core.GeminiAPI(throttle_rate=0)
    monkeypatch.setattr(api, "_rate_limit", lambda: None)
    response = Mock(status_code=200)
    response.json.return_value = {"candidates": [{"finishReason": "MAX_TOKENS",
                                                  "content": {"parts": [{"text": "half"}]}}]}
    monkeypatch.setattr(api, "_post_with_retry", lambda *a, **kw: response)
    with pytest.raises(core.ResponseError, match="MAX_TOKENS"):
        api.generate_text("prompt")


# ---------------------------------------------------------------- CLI

CONFIG = """
[models]
ranking = a, b, c
[model.a]
kind = groq
model = whisper-a
[model.b]
kind = groq
model = whisper-b
[model.c]
kind = groq
model = whisper-c
[fusion]
model = gemini-3.6-flash
"""


@pytest.fixture
def cli(tmp_path, monkeypatch):
    audio = tmp_path / "talk.wav"
    audio.touch()
    config = tmp_path / "t.ini"
    config.write_text(CONFIG)
    monkeypatch.setattr(runner, "is_reachable", lambda name, options: True)
    fake = transcriber(fail={"c"})
    monkeypatch.setattr(runner, "transcribe_with_model",
                        lambda name, options, audio, language, **kw: fake(audio, runner.model_spec(None, name, options)))
    seen = []
    monkeypatch.setattr(fusion, "gemini_fuser", lambda model, **kw: fuser(seen, "words from fused"))
    return audio, config, seen


def test_cli_fuse_writes_individual_and_fused_documents(cli, tmp_path, capsys):
    audio, config, seen = cli
    out = tmp_path / "out"
    out.mkdir()
    assert runner.main([str(audio), "--config", str(config), "-d", str(out), "--fuse", "a,b,c"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["talk.a.md", "talk.b.md", "talk.fused.md"]
    fused = (out / "talk.fused.md").read_text()
    assert fused.splitlines()[0] == f"src: {audio.resolve()}"
    assert "fusion: fused a, b with gemini-3.6-flash" in fused
    assert "fusion failure: c: ProviderError: c down" in fused
    assert fused.endswith("\n\nwords from fused\n")


def test_cli_fuse_needs_a_destination(cli):
    audio, config, _ = cli
    with pytest.raises(SystemExit):
        runner.main([str(audio), "--config", str(config), "--fuse", "a,b"])


def test_cli_fuse_rejects_unknown_models(cli, tmp_path):
    audio, config, _ = cli
    with pytest.raises(SystemExit):
        runner.main([str(audio), "--config", str(config), "-d", str(tmp_path), "--fuse", "a,zzz"])


# ---------------------------------------------------------------- evaluation

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_transcripts.py"


def test_scorer_compares_fusion_with_the_strongest_individual(tmp_path):
    spec = spec_from_file_location("score_transcripts", SCRIPT)
    scorer = module_from_spec(spec)
    spec.loader.exec_module(scorer)
    reference = tmp_path / "reference.txt"
    reference.write_text("one two three four five", encoding="utf-8")
    (tmp_path / "reference.local.json").write_text(json.dumps({
        "reference": "reference.txt", "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "origin": "human-from-audio", "status": "reviewed", "reviewed_by": "owner", "reviewed_on": "2026-10-14"}))
    files = {"a.txt": "one two three four", "b.txt": "one too six four", "fused.txt": "one two three four five"}
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    scorer.main(["--reference", str(reference), "--hypothesis", str(tmp_path / "a.txt"),
                 "--hypothesis", str(tmp_path / "b.txt"), "--fused", str(tmp_path / "fused.txt"),
                 "--output", str(tmp_path / "s.json")])
    report = json.loads((tmp_path / "s.json").read_text())
    assert report["fusion"]["best_individual"] == "a.txt"
    assert report["fusion"]["improved"] is True and report["fusion"]["wer_delta_percent"] == -20.0
