"""Loudness/noise assessment on every run; automatic enhancement into a separate file.

Owner decision (2026-10-14): enhancement is fully automatic when the assessment
finds a problem; --no-enhance (enhance=False) disables it. Signals are synthetic
speech-like tone bursts made with ffmpeg. BLIND SPOT: the effect of enhancement on
real transcription accuracy is not measured here.
"""
import hashlib
import shutil
import subprocess

import pytest

from multimodal_transcription_core import audio_quality, engine, runner
from multimodal_transcription_core.engine import ModelSpec

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")

BURSTS = "sine=f=300:d=10,volume='if(lt(mod(t,1),0.6),1,0)':eval=frame"


def make(path, gain=2.0, noise_db=None):
    if noise_db is None:
        command = ["-f", "lavfi", "-i", BURSTS, "-af", f"volume={gain}"]
    else:
        command = ["-f", "lavfi", "-i", BURSTS, "-f", "lavfi", "-i", "anoisesrc=d=10:c=white:a=1",
                   "-filter_complex", f"[0]volume={gain}[a];[1]volume={noise_db}dB[n];[a][n]amix=inputs=2:normalize=0"]
    subprocess.run(["ffmpeg", "-v", "error", *command, "-ar", "16000", "-ac", "1", "-y", str(path)], check=True)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("gain,noise_db,problems", [
    (2.0, None, []), (0.02, None, ["quiet"]), (12.0, None, ["loud"]), (2.0, -16, ["noisy"]),
])
def test_assessment_finds_quiet_loud_and_noisy_audio(tmp_path, gain, noise_db, problems):
    audio = make(tmp_path / "a.wav", gain, noise_db)
    assert audio_quality.assess(audio)["problems"] == problems


def test_assessment_never_modifies_or_adds_files(tmp_path):
    audio = make(tmp_path / "a.wav", 0.02)
    before = sha(audio)
    result = audio_quality.assess(audio)
    assert sha(audio) == before and [p.name for p in tmp_path.iterdir()] == ["a.wav"]
    assert {"integrated_lufs", "true_peak_dbtp", "snr_db", "noise_level_db"} <= set(result)


@pytest.mark.parametrize("gain,noise_db", [(0.02, None), (12.0, None), (2.0, -16)])
def test_enhancement_fixes_what_the_assessment_found(tmp_path, gain, noise_db):
    audio = make(tmp_path / "a.wav", gain, noise_db)
    before = audio_quality.assess(audio)
    enhanced = audio_quality.enhance(audio, before)
    after = audio_quality.assess(enhanced["path"])
    assert after["problems"] == [] and enhanced["filters"]
    assert enhanced["path"].parent == tmp_path and enhanced["path"].name.startswith("a.enhanced-")


def fake_request(seen):
    def request(spec, path, settings, clients):
        seen.append(path)
        return {"text": "hello", "no_speech": False}
    return request


SPEC = ModelSpec(name="p", kind="http", model="parakeet", endpoint="http://127.0.0.1:9/v1")


def test_problem_audio_is_enhanced_automatically_and_the_original_kept(tmp_path, monkeypatch):
    audio = make(tmp_path / "talk.wav", 0.02)
    before = sha(audio)
    seen = []
    monkeypatch.setattr(engine, "_request", fake_request(seen))
    result = engine.transcribe_file(audio, SPEC, timestamps="none")
    enhanced = result["audio"]["enhanced"]
    assert sha(audio) == before
    assert enhanced and seen[0].name == enhanced["path"].rsplit("/", 1)[-1]
    assert result["audio"]["transcribed"] == enhanced["path"]
    assert result["audio"]["assessment"]["problems"] == ["quiet"]
    again = engine.transcribe_file(audio, SPEC, timestamps="none")
    assert again["audio"]["enhanced"]["reused"] is True
    assert len(list(tmp_path.glob("talk.enhanced-*"))) == 1


def test_no_enhance_still_assesses_but_creates_nothing(tmp_path, monkeypatch):
    audio = make(tmp_path / "talk.wav", 0.02)
    seen = []
    monkeypatch.setattr(engine, "_request", fake_request(seen))
    result = engine.transcribe_file(audio, SPEC, timestamps="none", enhance=False)
    assert result["audio"]["enhanced"] is None and result["audio"]["assessment"]["problems"] == ["quiet"]
    assert seen[0] == audio and [p.name for p in tmp_path.iterdir()] == ["talk.wav"]


def test_good_audio_is_sent_unchanged(tmp_path, monkeypatch):
    audio = make(tmp_path / "talk.wav")
    seen = []
    monkeypatch.setattr(engine, "_request", fake_request(seen))
    result = engine.transcribe_file(audio, SPEC, timestamps="none")
    assert result["audio"]["enhanced"] is None and seen[0] == audio


def test_unwritable_source_folder_falls_back_to_the_original_with_a_warning(tmp_path, monkeypatch):
    audio = make(tmp_path / "talk.wav", 0.02)
    seen = []
    monkeypatch.setattr(engine, "_request", fake_request(seen))
    def refuse(*args, **kwargs):
        raise PermissionError("read-only")
    monkeypatch.setattr(audio_quality, "_render", refuse)
    result = engine.transcribe_file(audio, SPEC, timestamps="none")
    assert seen[0] == audio and result["audio"]["enhanced"] is None
    assert any("enhance" in w for w in result["warnings"])


def test_unreadable_audio_is_reported_not_fatal(tmp_path, monkeypatch):
    audio = tmp_path / "talk.wav"
    audio.write_bytes(b"not audio")
    monkeypatch.setattr(engine, "_request", fake_request([]))
    result = engine.transcribe_file(audio, SPEC, timestamps="none")
    assert result["audio"]["assessment"] is None
    assert any("loudness" in w for w in result["warnings"])


@pytest.mark.parametrize("flags,expected", [([], None), (["--no-enhance"], False), (["--enhance"], True)])
def test_cli_enhance_flag(tmp_path, monkeypatch, flags, expected):
    audio = tmp_path / "a.wav"
    audio.touch()
    config = tmp_path / "t.ini"
    config.write_text("[models]\nranking = w\n[model.w]\nkind = groq\n")
    seen = {}
    monkeypatch.setattr(engine, "transcribe_file", lambda audio, spec, **kw: seen.update(kw) or {"text": "t"})
    assert runner.main([str(audio), "--config", str(config), *flags]) == 0
    assert seen.get("enhance") == expected
