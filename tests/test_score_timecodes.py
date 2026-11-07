"""Offline checks of the time-code test: padding, clock slips, and the shift score."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import wave

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_timecodes.py"
spec = spec_from_file_location("score_timecodes", SCRIPT)
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_padded_copy_adds_known_silence(tmp_path):
    src, out = tmp_path / "a.wav", tmp_path / "b.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(b"\x01\x00" * 8000 * 10)
    assert module.padded_copy(src, out, cut=5.0) == 10.0
    with wave.open(str(out), "rb") as w:
        frames = w.readframes(w.getnframes())
        assert w.getnframes() == 8000 * 17
    assert frames[:2 * 8000 * 3] == b"\0" * 2 * 8000 * 3                     # 3 s pad
    assert frames[2 * 8000 * 8:2 * 8000 * 12] == b"\0" * 2 * 8000 * 4         # 4 s gap at 5 s


def test_clock_slips_are_read_as_seconds():
    assert module._seconds("12.5") == 12.5 and module._seconds("1:00.1") == 60.1


def word(w, s):
    return {"word": w, "start": s, "end": s + 0.3}


def test_real_aligner_scores_zero_and_text_times_do_not():
    original = [word("раз", 10.0), word("два", 70.0)]
    real = [word("раз", 13.0), word("два", 77.0)]
    fake = [word("раз", 10.5), word("два", 71.0)]
    assert module.score(original, real, None, 120.0, 60.0)["shift"]["median_error_s"] == 0.0
    row = module.score(original, fake, None, 120.0, 60.0)
    assert row["shift"]["median_error_s"] > 1.0 and row["shift"]["over_1s"] == 1.0


def test_times_past_the_end_of_audio_are_counted():
    assert module.score([word("раз", 158.0)], None, None, 120.0, 60.0)["past_end_of_audio"] == 1
