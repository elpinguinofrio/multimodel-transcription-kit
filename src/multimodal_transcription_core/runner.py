"""The cli-runner: one audio file in, one transcript out.

This module is the CLI layer, and it is the only place that reads a config
file. The importable library still receives every value explicitly, so
`import multimodal_transcription_core` discovers nothing.
"""

import argparse
import configparser
import json
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

from .errors import FusionError, NoModelAvailableError, TranscriptionError
from .rate_limiter import RateLimitError

DEFAULT_CONFIG_NAME = "transcription.ini"
EXAMPLE_CONFIG_NAME = "transcription.example.ini"
KEY_FOR_KIND = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}


def load_config(explicit):
    """Read the CLI's config file; an explicit path must exist."""
    path = Path(explicit).expanduser() if explicit else Path.cwd() / DEFAULT_CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No config at {path}. Copy {EXAMPLE_CONFIG_NAME} to {DEFAULT_CONFIG_NAME} and edit it.")
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    return parser


def load_secrets(config):
    """Hand the library one explicit env file; it never searches for one."""
    configured = config.get("secrets", "env_file", fallback=None)
    if not configured:
        return
    env = Path(configured).expanduser()
    if env.is_file():
        os.environ["TRANSCRIPTION_ENV_FILE"] = str(env)
        from .config import load_env_file
        load_env_file()


def ranked_models(config):
    """Model names best-first, as the owner ordered them in the config."""
    raw = config.get("models", "ranking", fallback="")
    names = [name.strip() for name in raw.split(",") if name.strip()]
    return [name for name in names if config.has_section(f"model.{name}")]


def model_options(config, name):
    section = f"model.{name}"
    if not config.has_section(section):
        return None
    return dict(config[section])


def is_reachable(name, options):
    """Can this model answer right now? Credentials for APIs, a socket for HTTP.

    BLIND SPOT: for API kinds this proves a key is present, not that it is valid;
    an expired key still reports reachable and fails at transcription time.
    """
    kind = options.get("kind")
    if kind == "http":
        url = urlparse(options.get("endpoint", ""))
        if not url.hostname:
            return False
        port = url.port or (443 if url.scheme == "https" else 80)
        try:
            socket.create_connection((url.hostname, port), timeout=1.5).close()
        except OSError:
            return False
        return True
    variable = options.get("requires_env") or KEY_FOR_KIND.get(kind)
    return bool(variable and os.environ.get(variable))


def model_spec(config, name, options):
    """The ModelSpec for a config section; [defaults] prompt applies to Gemini models
    whose own section sets none (it is written for Gemini's instruction prompt)."""
    from .engine import ModelSpec
    spec = ModelSpec.from_options(name, options)
    default_prompt = config.get("defaults", "prompt", fallback=None) if config is not None else None
    if spec.prompt is None and default_prompt and spec.kind == "gemini":
        from dataclasses import replace
        spec = replace(spec, prompt=default_prompt)
    return spec


def transcribe_with_model(name, options, audio_path, language, limiter=None, usage_log=None,
                          config=None, **settings):
    """Run the shared transcription path (engine.transcribe_file) for one configured model."""
    from .engine import transcribe_file
    return transcribe_file(audio_path, model_spec(config, name, options), language=language,
                           limiter=limiter, usage_log=usage_log, **settings)


def write_transcript(destination, audio, model, content):
    """Create `<source>.<model>.md`, else `(2)`, `(3)`... Exclusive creation, so
    simultaneous runs never overwrite each other or an earlier output."""
    base = f"{Path(audio).stem}.{model}"
    attempt = 1
    while True:
        candidate = destination / (f"{base}.md" if attempt == 1 else f"{base}.({attempt}).md")
        try:
            with open(candidate, "x", encoding="utf-8") as handle:
                handle.write(content)
            return candidate
        except FileExistsError:
            attempt += 1


def render_document(audio, result):
    """`src:` first, then what actually ran (model, features, settings, audio identity,
    fallbacks, warnings) as one line each, a blank line, and the transcript."""
    def line(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    header = [f"src: {Path(audio).resolve()}"]
    if result.get("model"):
        header.append(f"model: {result.get('name')} ({result.get('provider')} {result['model']})")
    for key in ("features", "settings", "audio"):
        if result.get(key):
            header.append(f"{key}: {line(result[key])}")
    if result.get("chunks", 1) > 1:
        header.append(f"chunks: {result['chunks']}")
    for attempt in (result.get("selection") or {}).get("attempts", []):
        if "error" in attempt:
            header.append(f"fallback: {attempt['name']}: {attempt['error']}")
        else:
            header.append(f"skipped: {attempt['name']}: {attempt['skipped']}")
    header.extend(f"warning: {warning}" for warning in result.get("warnings", []))
    return "\n".join(header) + f"\n\n{result['text'].strip()}\n"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="transcription-core",
        description="Transcribe one audio file. Use 'http --help' for a raw HTTP request.")
    parser.add_argument("audio", nargs="?", type=Path, help="Audio file to transcribe")
    parser.add_argument("-d", "--destination", type=Path,
                        help="Existing directory for the .md transcript; omit to print to stdout")
    parser.add_argument("-m", "--model", help="Model name from the config; omit to use the best reachable one")
    parser.add_argument("--language", help="Spoken language hint; defaults to the config value")
    parser.add_argument("--speakers", action=argparse.BooleanOptionalAction, default=None,
                        help="Speaker labels (default: on where the model supports them)")
    parser.add_argument("--sounds", action=argparse.BooleanOptionalAction, default=None,
                        help="Non-speech sounds such as *dog barking* (default: on where supported)")
    parser.add_argument("--timestamps", choices=("none", "major", "phrase"), default=None,
                        help="Timestamps from provider alignment (default: major where supported)")
    parser.add_argument("--config", help=f"Config file; defaults to ./{DEFAULT_CONFIG_NAME}")
    parser.add_argument("--enhance", action=argparse.BooleanOptionalAction, default=None,
                        help="Transcribe an enhanced temporary copy (default: automatic when loudness or noise "
                             "is a problem; --no-enhance never, --enhance always)")
    parser.add_argument("--keep-enhanced", action="store_true", default=None,
                        help="Keep the enhanced copy in [defaults] enhanced_dir, else ~/.cache/transcription-core/"
                             "enhanced (default: deleted after the run)")
    parser.add_argument("--fuse", metavar="MODELS",
                        help="Comma-separated configured models, or 'all' for the ranking: run each on the "
                             "same audio, keep every transcript, and combine them (needs -d)")
    parser.add_argument("--receipt", action="store_true",
                        help="With -d, also write <transcript>.receipt.json: spec, options, client "
                             "version, input hash and raw provider responses")
    parser.add_argument("--list-models", action="store_true", help="Show the ranking and what is reachable now")
    return parser


def _settings(args, config):
    """Options shared by every model in this run, from flags and [defaults]."""
    settings = {name: getattr(args, name) for name in ("speakers", "sounds", "timestamps", "enhance", "keep_enhanced")
                if getattr(args, name) is not None}
    overlap = config.get("defaults", "overlap_seconds", fallback=None)
    if overlap:
        settings["overlap_seconds"] = float(overlap)
    enhanced_dir = config.get("defaults", "enhanced_dir", fallback=None)
    if enhanced_dir:
        settings["enhanced_dir"] = str(Path(enhanced_dir).expanduser())
    return settings


def render_fused_document(audio, fused, written):
    """The fused transcript, naming its inputs (and their files), failures and fusion model."""
    record = fused["fusion"]
    if record["status"] == "fused":
        header = [f"fusion: fused {', '.join(record['inputs'])} with {record['model']}"]
    else:
        header = [f"fusion: not fused (only {record['inputs'][0]} succeeded)"]
    header += [f"fusion input: {name} -> {path.name}" for name, path in written]
    header += [f"fusion failure: {f['name']}: {f['error']}" for f in record["failures"]]
    return f"src: {Path(audio).resolve()}\n" + "\n".join(header) + f"\n\n{fused['text'].strip()}\n"


def _run_fusion(args, config, names, limiter, usage_log):
    from . import fusion
    language = args.language or config.get("defaults", "language", fallback=None)
    options_by_name = {name: model_options(config, name) for name in names}
    specs = [model_spec(config, name, options_by_name[name]) for name in names]
    model = config.get("fusion", "model", fallback=fusion.FUSION_MODEL)

    def run_model(audio, spec, limiter=None, **kwargs):
        return transcribe_with_model(spec.name, options_by_name[spec.name], audio, language,
                                     limiter=limiter, usage_log=usage_log, config=config, **kwargs)
    try:
        fused = fusion.transcribe_fused(
            args.audio, specs, fuse_texts=fusion.gemini_fuser(model, limiter=limiter, usage_log=usage_log),
            transcribe=run_model, probe=lambda spec: is_reachable(spec.name, options_by_name[spec.name]),
            limiter=limiter, fusion_model=model, **_settings(args, config))
        error = None
    except NoModelAvailableError as exc:
        print(f"transcription-core: {exc}", file=sys.stderr)
        return 1
    except FusionError as exc:
        fused, error = {"individual": exc.individual}, exc
    written = []
    for result in fused["individual"]:
        path = write_transcript(args.destination, args.audio, result["name"], render_document(args.audio, result))
        written.append((result["name"], path))
        print(path, file=sys.stderr)
    if error is not None:
        print(f"transcription-core: {error}; the individual transcripts are kept", file=sys.stderr)
        return 1
    path = write_transcript(args.destination, args.audio, "fused", render_fused_document(args.audio, fused, written))
    for failure in fused["fusion"]["failures"]:
        print(f"transcription-core: {failure['name']}: {failure['error']}", file=sys.stderr)
    print(path, file=sys.stderr)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if not args.list_models and args.audio is None:
        parser.error("Give an audio file, or --list-models")
    if args.audio is not None and not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    if args.destination is not None and not args.destination.is_dir():
        parser.error(f"Destination directory does not exist: {args.destination}")

    try:
        config = load_config(args.config)
    except FileNotFoundError as error:
        parser.error(str(error))
    load_secrets(config)
    from .rate_limiter import RateLimiter
    limiter = RateLimiter.from_parser(config)
    from .usage import UsageLog
    usage_log = UsageLog.from_parser(config)

    ranking = ranked_models(config)
    if not ranking:
        parser.error("No models configured; add a [models] ranking and one [model.<name>] section")

    if args.list_models:
        for name in ranking:
            options = model_options(config, name)
            state = "reachable" if is_reachable(name, options) else "unavailable"
            api_model = options.get("model", name)
            limits = limiter.limits.get(api_model)
            quota = ""
            if limits is not None and limits.rpd is not None:
                quota = f"  today {limiter.usage(api_model)['today']}/{limits.rpd}"
            from .selection import verify
            try:
                checked = verify(model_spec(config, name, options)).reason
            except ValueError as error:          # e.g. a section with an unknown kind
                checked = str(error)
            print(f"{name:<28} {options.get('kind', '?'):<8} {state}{quota}  {checked}")
        return 0

    if args.fuse:
        if args.destination is None:
            parser.error("--fuse needs -d, so every individual transcript is kept")
        if args.model:
            parser.error("Use either -m or --fuse")
        names = ranking if args.fuse.strip() == "all" else [n.strip() for n in args.fuse.split(",") if n.strip()]
        unknown = [name for name in names if model_options(config, name) is None]
        if unknown or len(names) < 2:
            parser.error(f"--fuse needs at least two configured models; unknown: {', '.join(unknown) or 'none'}")
        return _run_fusion(args, config, names, limiter, usage_log)

    if args.model:
        options = model_options(config, args.model)
        if options is None:
            parser.error(f"Unknown model {args.model!r}; known: {', '.join(ranking)}")
        if not is_reachable(args.model, options):
            parser.error(f"Model {args.model!r} is not reachable right now")
        candidates = [args.model]
    else:
        candidates = ranking
    options_by_name = {name: model_options(config, name) for name in candidates}
    specs = [model_spec(config, name, options_by_name[name]) for name in candidates]

    language = args.language or config.get("defaults", "language", fallback=None)
    settings = _settings(args, config)
    if args.receipt:
        if args.destination is None:
            parser.error("--receipt needs -d")
        settings["keep_raw"] = True

    def run_model(audio, spec, limiter=None, **kwargs):
        return transcribe_with_model(spec.name, options_by_name[spec.name], audio, language,
                                     limiter=limiter, usage_log=usage_log, config=config, **kwargs)

    from .selection import describe_error, transcribe_best
    try:
        result = transcribe_best(args.audio, specs, probe=lambda spec: is_reachable(spec.name, options_by_name[spec.name]),
                                 transcribe=run_model, limiter=limiter, allow_fallback=not args.model, **settings)
    except NoModelAvailableError as error:
        if not any("error" in attempt for attempt in error.attempts):
            parser.error(f"No configured model is available; check keys, endpoints and quota ({error})")
        print(f"transcription-core: {error}", file=sys.stderr)
        return 1
    except (TranscriptionError, RateLimitError) as error:
        print(f"transcription-core: {args.model}: {describe_error(error)}", file=sys.stderr)
        return 1
    chosen = result.get("selection", {}).get("chosen", candidates[0])
    for attempt in result.get("selection", {}).get("attempts", []):
        print(f"transcription-core: {attempt['name']}: {attempt.get('error') or attempt.get('skipped')}",
              file=sys.stderr)
    for warning in result.get("warnings", []):
        print(f"transcription-core: warning: {warning}", file=sys.stderr)
    if result.get("no_speech"):
        print("transcription-core: no speech detected", file=sys.stderr)

    if args.destination is None:
        print(result["text"].strip())
        return 0
    written = write_transcript(args.destination, args.audio, chosen, render_document(args.audio, result))
    print(written, file=sys.stderr)
    if args.receipt:
        from .receipts import build_receipt
        spec = next(s for s in specs if s.name == chosen)
        options = {"language": language, **{k: v for k, v in settings.items() if k != "keep_raw"}}
        receipt = build_receipt(result, spec, options, argv=list(sys.argv[1:] if argv is None else argv),
                                transcript_file=written.name)
        path = written.with_name(written.name[:-len(".md")] + ".receipt.json")
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
