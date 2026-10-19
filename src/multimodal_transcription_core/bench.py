"""Benchmark runs: numbered transcripts plus receipts that reproduce them.

    transcription-core bench --config C --audio A --runs-dir D -m NAME [-m NAME ...] --label L
    transcription-core bench --replay D/_details/receipts/<run>.json --runs-dir D

Each attempt writes ``<model>__<label>__v###.txt`` (successes only) and always a
receipt ``_details/receipts/<same>.json``: model spec, options, prompt and
generation settings sent, client version, input hash, raw provider responses and
any error. Numbers are claimed with exclusive creation; nothing is overwritten.
Repeats are interleaved across models.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

FEATURES = ("speakers", "sounds", "timestamps")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _claim(runs, base):
    """Next free version: the receipt file is created exclusively as the claim."""
    receipts = runs / "_details" / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        stem = f"{base}__v{number:03d}"
        if not (runs / f"{stem}.txt").exists():
            try:
                return stem, open(receipts / f"{stem}.json", "x", encoding="utf-8")
            except FileExistsError:
                pass
        number += 1


def run_one(audio, spec, options, *, runs, label, argv, limiter=None, usage_log=None, replay_of=None):
    """One attempt; returns (ok, transcript file name, plain text). Failures are recorded, never discarded."""
    from .engine import transcribe_file
    from .errors import TranscriptionError
    from .rate_limiter import RateLimitError
    from .receipts import build_receipt
    from .selection import describe_error
    stem, handle = _claim(runs, f"{spec.model.replace('/', '_')}__{label}")
    with handle:
        try:
            result = transcribe_file(audio, spec, limiter=limiter, usage_log=usage_log, keep_raw=True, **options)
            status, error = "ok", None
        except (TranscriptionError, RateLimitError, OSError, ValueError) as exc:
            result = getattr(exc, "result", None) or {
                "audio": {"source": str(Path(audio).resolve()), "sha256": _sha256(audio)}}
            status, error = "failed", describe_error(exc)
        transcript = f"{stem}.txt" if status == "ok" else None
        if transcript:
            with open(runs / transcript, "x", encoding="utf-8") as out:
                out.write(result["text"].strip() + "\n")
        receipt = build_receipt(result, spec, options, status=status, error=error, label=label,
                                argv=list(argv), transcript_file=transcript, replay_of=replay_of)
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    line = f"{stem}: {status}" + (f" ({error})" if error else "")
    print(line, file=sys.stderr if error else sys.stdout)
    return status == "ok", transcript, result.get("plain_text", result.get("text"))


def fuse_round(round_results, *, runs, label, argv, fuse_texts, model):
    """Fuse one round's successful transcripts into fused__<label>__v###.txt with a receipt."""
    from .fusion import _check
    from .receipts import client_info
    from .selection import describe_error
    inputs = [(name, text) for ok, name, text in round_results if ok]
    failures = [name for ok, name, _ in round_results if not ok]
    stem, handle = _claim(runs, f"fused__{label}")
    status, error, transcript = "not fused", None, None
    if len(inputs) >= 2:
        try:
            fused = (fuse_texts([text for _, text in inputs]) or "").strip()
            error = _check(fused, [text for _, text in inputs]) or None
        except Exception as exc:              # recorded in the receipt, never discarded
            fused, error = "", describe_error(exc)
        status = "failed" if error else "ok"
        if not error:
            transcript = f"{stem}.txt"
            with open(runs / transcript, "x", encoding="utf-8") as out:
                out.write(fused + "\n")
    with handle:
        json.dump({"schema": 1, "client": client_info(), "label": label, "argv": list(argv),
                   "status": status, "error": error, "transcript_file": transcript,
                   "fusion": {"inputs": [name for name, _ in inputs], "failed_members": failures,
                              "model": model}}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"{stem}: {status}" + (f" ({error})" if error else ""))
    return status == "ok"


def build_parser():
    parser = argparse.ArgumentParser(prog="transcription-core bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Config with [model.<name>] sections, limits, usage and secrets")
    parser.add_argument("--audio", type=Path, help="Approved benchmark recording")
    parser.add_argument("--runs-dir", type=Path, required=True, help="Where transcripts and receipts go")
    parser.add_argument("-m", "--model", action="append", default=[], help="Configured model name; repeatable")
    parser.add_argument("--label", default="run", help="Setup label in file names, e.g. p1-verbatim")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per model, interleaved")
    parser.add_argument("--language")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--speakers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sounds", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--timestamps", choices=("none", "major", "phrase"))
    parser.add_argument("--fuse", action="store_true",
                        help="After each round, fuse the models' transcripts into fused__<label>__v###.txt "
                             "(score it with scripts/score_transcripts.py --fused)")
    parser.add_argument("--replay", type=Path, help="Re-run exactly the setup recorded in this receipt")
    return parser


def _environment(config):
    """Limiter, usage log and secrets from an explicit config, if one was given."""
    if config is None:
        return None, None
    from .rate_limiter import RateLimiter
    from .runner import load_secrets
    from .usage import UsageLog
    load_secrets(config)
    return RateLimiter.from_parser(config), UsageLog.from_parser(config)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    from .engine import ModelSpec
    from .runner import load_config, model_options, model_spec
    config = load_config(args.config) if args.config else None
    limiter, usage_log = _environment(config)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    if args.replay:
        receipt = json.loads(args.replay.read_text(encoding="utf-8"))
        spec_fields = dict(receipt["spec"])
        if spec_fields.get("capabilities") is not None:
            spec_fields["capabilities"] = frozenset(spec_fields["capabilities"])
        spec = ModelSpec(**spec_fields)
        audio = Path(args.audio or receipt["audio"]["source"])
        if not audio.is_file() or _sha256(audio) != receipt["audio"]["sha256"]:
            parser.error(f"{audio} is not the recording this receipt was made from (sha256 differs)")
        ok, _, _ = run_one(audio, spec, receipt["options"], runs=args.runs_dir, label=receipt["label"],
                           argv=argv, limiter=limiter, usage_log=usage_log, replay_of=str(args.replay.resolve()))
        return 0 if ok else 1

    if config is None or args.audio is None or not args.model:
        parser.error("Give --config, --audio and at least one -m, or --replay")
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    # The same [defaults] the main CLI applies, so a benchmark measures what users run.
    options = {"language": args.language or config.get("defaults", "language", fallback=None)}
    overlap = config.get("defaults", "overlap_seconds", fallback=None)
    if overlap:
        options["overlap_seconds"] = float(overlap)
    enhanced_dir = config.get("defaults", "enhanced_dir", fallback=None)
    if enhanced_dir:
        options["enhanced_dir"] = str(Path(enhanced_dir).expanduser())
    if args.temperature is not None:
        options["temperature"] = args.temperature
    options.update({name: getattr(args, name) for name in FEATURES if getattr(args, name) is not None})
    specs = []
    for name in args.model:
        section = model_options(config, name)
        if section is None:
            parser.error(f"Unknown model {name!r}")
        specs.append(model_spec(config, name, section))
    if args.fuse and len(specs) < 2:
        parser.error("--fuse needs at least two -m models")
    if args.fuse:
        from . import fusion
        fusion_model = config.get("fusion", "model", fallback=fusion.FUSION_MODEL)
        fuse_texts = fusion.gemini_fuser(fusion_model, limiter=limiter, usage_log=usage_log)
    outcomes = []
    for _ in range(args.repeat):
        round_results = [run_one(args.audio, spec, options, runs=args.runs_dir, label=args.label, argv=argv,
                                 limiter=limiter, usage_log=usage_log) for spec in specs]
        outcomes += [ok for ok, _, _ in round_results]
        if args.fuse:
            outcomes.append(fuse_round(round_results, runs=args.runs_dir, label=args.label, argv=argv,
                                       fuse_texts=fuse_texts, model=fusion_model))
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
