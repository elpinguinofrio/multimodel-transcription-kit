"""Loudness/noise assessment on every run; automatic enhancement into a temporary file.

Owner decision (2026-10-14): enhancement is fully automatic when the assessment
finds a problem; --no-enhance (enhance=False) disables it. Owner decision
(2026-10-16): the enhanced copy is an intermediate, deleted after the run unless
the caller asks to keep it (keep_enhanced=True / --keep-enhanced). Signals are synthetic
speech-like tone bursts made with ffmpeg. BLIND SPOT: the effect of enhancement on
real transcription accuracy is not measured here.
"""
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from multimodal_transcription_core import audio_quality, engine, runner
from multimodal_transcription_core.engine import ModelSpec

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")

BURSTS = "sine=f=300:d=10,volume='if(lt(mod(t,1),0.6),1,0)':eval=frame"
# Moderate speech-like bursts with one 10 ms spike near full scale: loud only by true peak.
PEAKY = "aevalsrc='0.06*sin(2*PI*300*t)*lt(mod(t,1),0.6)+0.95*sin(2*PI*1000*t)*between(t,5,5.01)':d=10:s=16000"


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
    assert enhanced["path"].parent == audio_quality.default_cache_dir()
    assert enhanced["path"].name.startswith("a.enhanced-") and [p.name for p in tmp_path.iterdir()] == ["a.wav"]


def fake_request(seen):
    def request(spec, path, settings, clients):
        seen.append(path)
        return {"text": "hello", "no_speech": False}
    return request


SPEC = ModelSpec(name="p", kind="http", model="parakeet", endpoint="http://127.0.0.1:9/v1")


@pytest.fixture
def places(tmp_path, monkeypatch):
    """Every folder a run could write to, all under tmp_path: source, cache, home and temp."""
    folders = {name: tmp_path / name for name in ("src", "xdg", "home", "tmp")}
    for folder in folders.values():
        folder.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(folders["xdg"]))
    monkeypatch.setenv("HOME", str(folders["home"]))
    monkeypatch.setattr(tempfile, "tempdir", str(folders["tmp"]))
    return folders


def files_left(folders):
    root = folders["src"].parent
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_problem_audio_is_enhanced_transcribed_and_then_deleted(places, monkeypatch):
    audio = make(places["src"] / "talk.wav", 0.02)
    before = sha(audio)
    seen = []

    def request(spec, path, settings, clients):
        seen.append((path, path.is_file()))
        return {"text": "hello", "no_speech": False}
    monkeypatch.setattr(engine, "_request", request)
    result = engine.transcribe_file(audio, SPEC, timestamps="none")
    enhanced = result["audio"]["enhanced"]
    assert sha(audio) == before and result["audio"]["assessment"]["problems"] == ["quiet"]
    assert seen[0] == (Path(enhanced["path"]), True)       # the copy existed while it was transcribed
    assert result["audio"]["transcribed"] == enhanced["path"]
    assert enhanced["kept"] is False and enhanced["sha256"] and enhanced["filters"]
    assert files_left(places) == ["src/talk.wav"]           # nothing left anywhere after the run


def test_enhanced_copy_is_deleted_when_the_run_fails(places, monkeypatch):
    audio = make(places["src"] / "talk.wav", 0.02)

    def fail(spec, path, settings, clients):
        raise engine.UnsupportedFeatureError("provider refused")
    monkeypatch.setattr(engine, "_request", fail)
    with pytest.raises(engine.UnsupportedFeatureError):
        engine.transcribe_file(audio, SPEC, timestamps="none")
    assert files_left(places) == ["src/talk.wav"]


def test_keep_enhanced_keeps_the_copy_in_the_named_folder(places, monkeypatch):
    audio = make(places["src"] / "talk.wav", 0.02)
    monkeypatch.setattr(engine, "_request", fake_request([]))
    result = engine.transcribe_file(audio, SPEC, timestamps="none", keep_enhanced=True,
                                    enhanced_dir=places["home"] / "kept")
    enhanced = result["audio"]["enhanced"]
    assert enhanced["kept"] is True and Path(enhanced["path"]).parent == places["home"] / "kept"
    assert files_left(places) == ["home/kept/" + Path(enhanced["path"]).name, "src/talk.wav"]


def test_kept_copies_default_to_the_cache_folder_and_are_reused(places, monkeypatch):
    audio = make(places["src"] / "talk.wav", 0.02)
    monkeypatch.setattr(engine, "_request", fake_request([]))
    first = engine.transcribe_file(audio, SPEC, timestamps="none", keep_enhanced=True)
    again = engine.transcribe_file(audio, SPEC, timestamps="none", keep_enhanced=True)
    assert Path(first["audio"]["enhanced"]["path"]).parent == audio_quality.default_cache_dir()
    assert again["audio"]["enhanced"]["reused"] is True
    assert len(list(audio_quality.default_cache_dir().glob("talk.enhanced-*"))) == 1


def test_enhanced_dir_alone_does_not_keep_the_copy(places, monkeypatch):
    audio = make(places["src"] / "talk.wav", 0.02)
    monkeypatch.setattr(engine, "_request", fake_request([]))
    result = engine.transcribe_file(audio, SPEC, timestamps="none", enhanced_dir=places["home"] / "kept")
    assert result["audio"]["enhanced"]["kept"] is False and files_left(places) == ["src/talk.wav"]


def test_enhanced_copy_is_16_bit(tmp_path):
    audio = make(tmp_path / "a.wav", 0.02)
    enhanced = audio_quality.enhance(audio, audio_quality.assess(audio), cache_dir=tmp_path / "c")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_fmt", "-of",
                            "default=nw=1", str(enhanced["path"])], capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "sample_fmt=s16"


def test_loud_peaks_alone_are_limited_not_made_louder(tmp_path):
    audio = tmp_path / "peaky.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", PEAKY, "-ac", "1", "-y", str(audio)], check=True)
    before = audio_quality.assess(audio)
    assert before["problems"] == ["loud"] and before["integrated_lufs"] < audio_quality.LOUD_LUFS
    after = audio_quality.assess(audio_quality.enhance(audio, before, cache_dir=tmp_path / "c")["path"])
    assert after["problems"] == [] and abs(after["integrated_lufs"] - before["integrated_lufs"]) < 2


def test_default_cache_folder_is_xdg_cache_home_then_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert audio_quality.default_cache_dir() == tmp_path / "xdg" / "transcription-core" / "enhanced"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert audio_quality.default_cache_dir() == tmp_path / ".cache" / "transcription-core" / "enhanced"


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


@pytest.mark.parametrize("flags,expected", [([], None), (["--keep-enhanced"], True)])
def test_cli_keep_enhanced_flag(tmp_path, monkeypatch, flags, expected):
    audio = tmp_path / "a.wav"
    audio.touch()
    config = tmp_path / "t.ini"
    config.write_text("[models]\nranking = w\n[model.w]\nkind = groq\n")
    seen = {}
    monkeypatch.setattr(engine, "transcribe_file", lambda audio, spec, **kw: seen.update(kw) or {"text": "t"})
    assert runner.main([str(audio), "--config", str(config), *flags]) == 0
    assert seen.get("keep_enhanced") == expected


def test_cli_config_names_the_enhanced_folder(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.touch()
    config = tmp_path / "t.ini"
    config.write_text("[defaults]\nenhanced_dir = ~/enhanced\n[models]\nranking = w\n[model.w]\nkind = groq\n")
    seen = {}
    monkeypatch.setattr(engine, "transcribe_file", lambda audio, spec, **kw: seen.update(kw) or {"text": "t"})
    assert runner.main([str(audio), "--config", str(config)]) == 0
    assert seen.get("enhanced_dir") == str(Path("~/enhanced").expanduser())
