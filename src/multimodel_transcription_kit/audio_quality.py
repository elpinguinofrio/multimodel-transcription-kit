"""Loudness and noise assessment, and enhancement into a separate file (ffmpeg only).

assess() reads the audio once and changes nothing. Speech and noise levels are
the 95th and 10th percentiles of 50 ms window RMS (digital silence counts as
very low); their difference estimates the signal-to-noise ratio. On synthetic
speech-like bursts this estimate ran about 5 dB above the true SNR.

enhance() writes a 16-bit `<stem>.enhanced-<id>.flac` into the caller's cache_dir
(the engine passes its per-run temporary folder unless the caller keeps copies),
else $XDG_CACHE_HOME/multimodel-transcription-kit/enhanced, else ~/.cache/multimodel-transcription-kit/enhanced;
never beside the source. Noisy audio gets a high-pass and FFT denoise. Loudness is
normalised (EBU R128) when it is too quiet or too loud; audio that is loud only by
its true peak is peak-limited at its own level instead. The id hashes the source
and the filter chain, so the same input reuses the same file; the original is never
modified.
"""

import hashlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path

from .errors import AudioPreparationError

QUIET_LUFS = -35.0          # integrated loudness below this: too quiet
LOUD_LUFS = -12.0           # above this: too loud
MAX_TRUE_PEAK_DBTP = -1.0   # above this: clipping risk
NOISY_SNR_DB = 15.0         # estimated SNR below this: noisy (about 10 dB true SNR)
MIN_WINDOWS = 20            # under 1 s of audio the noise estimate is not attempted
SILENCE_DB = -120.0
TARGET = "loudnorm=I=-16:TP=-1.5:LRA=11"
PEAK_LIMIT = ["aresample=16000", "alimiter=limit=0.79:level=0"]   # about -2 dBFS; level=0 keeps loudness
DENOISE_STRENGTH_DB = 12
_ANALYSIS = ("aresample=16000,aformat=channel_layouts=mono,asetnsamples=n=800,"
             "astats=metadata=1:reset=1:measure_perchannel=none:measure_overall=RMS_level,"
             "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-,loudnorm=print_format=json")


def _ffmpeg(args):
    try:
        return subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", *args], capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioPreparationError("ffmpeg is required to assess loudness") from exc


def assess(path):
    """Measure loudness and noise without modifying anything; list detected problems."""
    run = _ffmpeg(["-i", str(path), "-vn", "-af", _ANALYSIS, "-f", "null", "-"])
    if run.returncode or "{" not in run.stderr:
        raise AudioPreparationError(f"ffmpeg could not read {Path(path).name} for assessment")
    stats = json.loads(run.stderr[run.stderr.rindex("{"):run.stderr.rindex("}") + 1])
    levels = sorted(SILENCE_DB if value == "-inf" else max(SILENCE_DB, float(value))
                    for value in re.findall(r"RMS_level=(\S+)", run.stdout))

    def number(key):
        try:
            return float(stats[key])
        except (KeyError, ValueError):
            return None
    integrated, peak = number("input_i"), number("input_tp")
    speech = levels[int(0.95 * (len(levels) - 1))] if levels else None
    noise = levels[int(0.10 * (len(levels) - 1))] if levels else None
    snr = speech - noise if levels and len(levels) >= MIN_WINDOWS else None
    problems = []
    if integrated is not None and integrated < QUIET_LUFS:
        problems.append("quiet")
    if (integrated is not None and integrated > LOUD_LUFS) or (peak is not None and peak > MAX_TRUE_PEAK_DBTP):
        problems.append("loud")
    if snr is not None and snr < NOISY_SNR_DB:
        problems.append("noisy")
    return {"integrated_lufs": integrated, "true_peak_dbtp": peak, "speech_level_db": speech,
            "noise_level_db": noise, "snr_db": None if snr is None else round(snr, 1),
            "problems": problems,
            "thresholds": {"quiet_lufs": QUIET_LUFS, "loud_lufs": LOUD_LUFS,
                           "max_true_peak_dbtp": MAX_TRUE_PEAK_DBTP, "noisy_snr_db": NOISY_SNR_DB}}


def filters_for(assessment):
    """ffmpeg filters that address the assessment's problems (empty when there are none)."""
    if not assessment["problems"]:
        return []
    chain = []
    if "noisy" in assessment["problems"]:
        floor = max(-80, min(-20, round(assessment["noise_level_db"])))
        chain += ["highpass=f=80", f"afftdn=nr={DENOISE_STRENGTH_DB}:nf={floor}"]
    integrated = assessment.get("integrated_lufs")
    peak_only = ("loud" in assessment["problems"] and "quiet" not in assessment["problems"]
                 and integrated is not None and integrated <= LOUD_LUFS)
    return chain + (PEAK_LIMIT if peak_only else [TARGET])


def _render(source, target, chain):
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    run = _ffmpeg(["-v", "error", "-i", str(source), "-vn", "-af", ",".join(chain),
                   "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", "-c:a", "flac", "-f", "flac", str(partial)])
    if run.returncode:
        partial.unlink(missing_ok=True)
        raise AudioPreparationError(f"ffmpeg enhancement failed: {run.stderr.strip()[-300:]}")
    if target.exists():                      # another run made the identical file first
        partial.unlink()
    else:
        os.replace(partial, target)


def default_cache_dir():
    """Where enhanced copies go when the caller names no folder (resolved at call time)."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "multimodel-transcription-kit" / "enhanced"


def enhance(source, assessment, *, force=False, cache_dir=None):
    """Write the enhanced copy into ``cache_dir`` (default: default_cache_dir()), or reuse it.

    force=True normalises loudness even when no problem was found.
    """
    source = Path(source)
    chain = filters_for(assessment) or ([TARGET] if force else [])
    if not chain:
        return None
    digest = hashlib.sha256()
    with open(source, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    digest.update(("|".join(chain) + "|16000|mono|s16|flac").encode())
    folder = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{source.stem}.enhanced-{digest.hexdigest()[:12]}.flac"
    reused = target.exists()
    if not reused:
        _render(source, target, chain)
    return {"path": target, "filters": chain, "reused": reused,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
