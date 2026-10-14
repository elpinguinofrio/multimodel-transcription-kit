"""The cli-runner: one audio file in, one transcript out.

This module is the CLI layer, and it is the only place that reads a config
file. The importable library still receives every value explicitly, so
`import multimodal_transcription_core` discovers nothing.
"""

import argparse
import configparser
import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

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


def transcribe_with_model(name, options, audio_path, language, limiter=None, usage_log=None):
    """Dispatch to the client for this model's kind and return transcript text."""
    kind = options.get("kind")
    api_model = options.get("model", name)
    usage_options = {"usage_log": usage_log} if usage_log is not None else {}
    if kind == "gemini":
        from .gemini_client import GeminiAPI
        return GeminiAPI(rate_limiter=limiter, **usage_options).speech_to_text(str(audio_path), model=api_model, language=language)["text"]
    if kind == "groq":
        from .groq_client import STTAPI, STTModel
        result = STTAPI(rate_limiter=limiter, **usage_options).speech_to_text(str(audio_path), model=STTModel(api_model),
                                         language=language, response_format="verbose_json")
        return result["text"]
    if kind == "http":
        from .http_client import run as transcribe_http
        work = Path(tempfile.mkdtemp(prefix="transcription-core-")) / "run"
        try:
            transcribe_http(Path(audio_path), options["endpoint"], work,
                            model=api_model, language=language, timeout=3600, **usage_options)
            return json.loads((work / "response.json").read_text(encoding="utf-8"))["text"]
        finally:
            shutil.rmtree(work.parent, ignore_errors=True)
    raise ValueError(f"Unknown provider kind for model {name}: {kind!r}")


def transcript_path(destination, audio, model):
    """`<source>.<model>.md`, then `(2)`, `(3)`; an existing file is never reused."""
    base = f"{Path(audio).stem}.{model}"
    candidate = destination / f"{base}.md"
    attempt = 1
    while candidate.exists():
        attempt += 1
        candidate = destination / f"{base}.({attempt}).md"
    return candidate


def render_document(audio, text):
    """The transcript document: source path, blank line, transcription."""
    return f"src: {Path(audio).resolve()}\n\n{text.strip()}\n"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="transcription-core",
        description="Transcribe one audio file. Use 'http --help' for a raw HTTP request.")
    parser.add_argument("audio", nargs="?", type=Path, help="Audio file to transcribe")
    parser.add_argument("-d", "--destination", type=Path,
                        help="Existing directory for the .md transcript; omit to print to stdout")
    parser.add_argument("-m", "--model", help="Model name from the config; omit to use the best reachable one")
    parser.add_argument("--language", help="Spoken language hint; defaults to the config value")
    parser.add_argument("--config", help=f"Config file; defaults to ./{DEFAULT_CONFIG_NAME}")
    parser.add_argument("--list-models", action="store_true", help="Show the ranking and what is reachable now")
    return parser


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
            print(f"{name:<28} {options.get('kind', '?'):<8} {state}{quota}")
        return 0

    if args.model:
        options = model_options(config, args.model)
        if options is None:
            parser.error(f"Unknown model {args.model!r}; known: {', '.join(ranking)}")
        if not is_reachable(args.model, options):
            parser.error(f"Model {args.model!r} is not reachable right now")
        chosen = args.model
    else:
        chosen = next((name for name in ranking if is_reachable(name, model_options(config, name))), None)
        if chosen is None:
            parser.error("No configured model is reachable; check keys and local endpoints")
        options = model_options(config, chosen)

    language = args.language or config.get("defaults", "language", fallback=None)
    usage_options = {"usage_log": usage_log} if usage_log is not None else {}
    text = transcribe_with_model(chosen, options, args.audio, language, limiter=limiter, **usage_options)

    if args.destination is None:
        print(text.strip())
        return 0
    written = transcript_path(args.destination, args.audio, chosen)
    written.write_text(render_document(args.audio, text), encoding="utf-8")
    print(written, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
