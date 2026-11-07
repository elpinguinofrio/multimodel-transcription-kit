"""Test whether models give real word time codes, without a timed reference.

Shift test: a copy of the WAV gets PAD seconds of silence at the start and GAP more
at CUT seconds. A model that aligns to the audio moves every matched word by exactly
PAD (before CUT) or PAD+GAP (after); a model that writes times as text does not.
Agreement: median |start difference| against --reference-model on the original
(agreement between models, not accuracy).

Timestamp-capable models (groq, http) run through the kit with phrase timestamps.
Gemini models are asked for a JSON word list with times (their only route).
The padded copy lives in a temporary folder deleted on exit. Writes only --output.

    python scripts/score_timecodes.py SAMPLE.wav --config transcription.ini \
        --model parakeet-tdt-0.6b-v3 --model gemini-3.6-flash --output timecodes.json
"""

import argparse
import json
import re
import statistics
import sys
import tempfile
import time
import wave
from difflib import SequenceMatcher
from pathlib import Path

from multimodel_transcription_kit import runner
from multimodel_transcription_kit.alignment import _norm, timed_words

PAD, GAP, CUT, EXCLUDE = 3.0, 4.0, 60.0, 1.0
GEMINI_PROMPT = (
    "Transcribe this audio verbatim, word by word, in the language spoken (English brand names in "
    "Latin script). For EVERY spoken word give its start and end time in seconds from the beginning "
    "of the audio, as precisely as you can. Output only a JSON array, no prose, like "
    '[{"w":"word","s":0.00,"e":0.31}, ...].')
_ITEM = re.compile(r'"w"\s*:\s*"([^"]+)"\s*,\s*"s"\s*:\s*([\d.:]+)\s*,\s*"e"\s*:\s*([\d.:]+)')


def _seconds(value):
    """Seconds from "12.3" or a clock slip like "1:00.1"."""
    parts = value.split(":")
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))


def padded_copy(source, target, cut=CUT):
    with wave.open(str(source), "rb") as src:
        params, frames = src.getparams(), src.readframes(src.getnframes())
    width = params.sampwidth * params.nchannels
    at = int(min(cut, params.nframes / params.framerate) * params.framerate) * width
    silence = lambda s: b"\0" * int(s * params.framerate) * width
    with wave.open(str(target), "wb") as out:
        out.setparams(params)
        out.writeframes(silence(PAD) + frames[:at] + silence(GAP) + frames[at:])
    return params.nframes / params.framerate


def gemini_words(name, options, audio, limiter, usage_log):
    from multimodel_transcription_kit.gemini_client import GeminiAPI
    api = GeminiAPI(rate_limiter=limiter, usage_log=usage_log, max_retries=1)
    text = api.speech_to_text(str(audio), model=options.get("model", name), full_prompt=GEMINI_PROMPT,
                              temperature=0.0).get("text") or ""
    words = [{"word": w, "start": _seconds(s), "end": _seconds(e)} for w, s, e in _ITEM.findall(text)]
    if not words:
        raise ValueError(f"no parsable word times in the reply: {text[:120]!r}")
    return words


def model_words(name, options, audio, config, limiter, usage_log):
    if options.get("kind") == "gemini":
        return gemini_words(name, options, audio, limiter, usage_log)
    result = runner.transcribe_with_model(name, options, str(audio), None, limiter=limiter, usage_log=usage_log,
                                          config=config, timestamps="phrase", speakers=False, sounds=False,
                                          keep_raw=True, enhance=False)
    return timed_words(result)


def matched(a, b):
    m = SequenceMatcher(None, [_norm(w["word"]) for w in a], [_norm(w["word"]) for w in b], autojunk=False)
    for block in m.get_matching_blocks():
        for k in range(block.size):
            yield a[block.a + k], b[block.b + k]


def score(original, padded, reference, duration, cut):
    row = {"words": len(original), "last_end": round(max(w["end"] for w in original), 2),
           "past_end_of_audio": sum(w["end"] > duration + 0.5 for w in original)}
    if padded:
        errors = [abs((p["start"] - o["start"]) - (PAD if o["start"] < cut else PAD + GAP))
                  for o, p in matched(original, padded) if abs(o["start"] - cut) >= EXCLUDE]
        if errors:
            row["shift"] = {"matched": len(errors), "median_error_s": round(statistics.median(errors), 3),
                            "within_0_25s": round(sum(e <= 0.25 for e in errors) / len(errors), 3),
                            "over_1s": round(sum(e > 1.0 for e in errors) / len(errors), 3)}
    if reference is not None and reference is not original:
        diffs = [abs(a["start"] - b["start"]) for a, b in matched(original, reference)]
        if diffs:
            row["agreement"] = {"matched": len(diffs), "median_s": round(statistics.median(diffs), 3)}
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path, help="PCM WAV sample")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", action="append", required=True, help="Configured model; repeat")
    parser.add_argument("--reference-model", help="Agreement reference (default: first --model)")
    parser.add_argument("--cut", type=float, default=CUT, help="Where the extra silence goes (seconds)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = runner.load_config(args.config)
    runner.load_secrets(config)
    from multimodel_transcription_kit.rate_limiter import RateLimiter
    from multimodel_transcription_kit.usage import UsageLog
    limiter, usage_log = RateLimiter.from_parser(config), UsageLog.from_parser(config)
    unknown = [m for m in args.model if runner.model_options(config, m) is None]
    if unknown:
        parser.error(f"not configured: {', '.join(unknown)}")
    cut = args.cut
    runs, errors = {}, {}
    with tempfile.TemporaryDirectory(prefix="timecodes-") as tmp:
        padded = Path(tmp) / f"{args.audio.stem}.padded.wav"
        duration = padded_copy(args.audio, padded, cut)
        for name in args.model:
            options = runner.model_options(config, name)
            for label, audio in (("original", args.audio), ("padded", padded)):
                try:
                    runs.setdefault(name, {})[label] = model_words(name, options, audio, config, limiter, usage_log)
                except Exception as error:                   # a model's failure is a result here
                    errors.setdefault(name, {})[label] = f"{type(error).__name__}: {error}"[:300]
                print(f"{name} {label}: {len(runs.get(name, {}).get(label) or []) or errors[name][label]}",
                      file=sys.stderr)
    reference_name = args.reference_model or args.model[0]
    reference = runs.get(reference_name, {}).get("original")
    results = {}
    for name in args.model:
        original = runs.get(name, {}).get("original")
        results[name] = ({"errors": errors.get(name, {})} if not original else
                         {**score(original, runs[name].get("padded"), reference, duration, cut),
                          "errors": errors.get(name, {})})
    report = {"audio": str(args.audio.resolve()), "duration_s": round(duration, 3),
              "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "method": {"pad_s": PAD, "gap_s": GAP, "cut_s": cut, "excluded_around_cut_s": EXCLUDE,
                         "agreement_reference": reference_name, "gemini_prompt": GEMINI_PROMPT},
              "results": results, "words": runs}
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    for name, row in results.items():
        shift = row.get("shift", {})
        print(f"{name:<26} words={row.get('words', 0):>4} last_end={row.get('last_end', '-')!s:>7} "
              f"shift_median={shift.get('median_error_s', '-')!s:>6} within_0.25s={shift.get('within_0_25s', '-')!s:>5} "
              f"over_1s={shift.get('over_1s', '-')!s:>5} agree_median={row.get('agreement', {}).get('median_s', '-')!s:>6} "
              f"errors={len(row['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
