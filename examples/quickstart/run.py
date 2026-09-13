#!/usr/bin/env python3
"""Transcribe one recording with every provider enabled in config.ini.

Shows the intended split: this application owns configuration and secrets;
multimodal_transcription_core receives explicit values and reads no files
of its own.

    python run.py                 # uses ./config.ini
    python run.py other.ini       # uses another config
"""
import configparser
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def load_config(path: Path) -> configparser.ConfigParser:
    if not path.is_file():
        sys.exit(f"No config at {path}. Copy config.ini and edit it.")
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def load_secrets(cfg: configparser.ConfigParser) -> None:
    """Point the library at an explicit .env - it never searches for one."""
    configured = cfg["secrets"]["env_file"] if cfg.has_option("secrets", "env_file") else None
    env = Path(configured).expanduser() if configured else HERE / ".env"
    if env.is_file():
        os.environ["TRANSCRIPTION_ENV_FILE"] = str(env)
        from multimodal_transcription_core.config import load_env_file
        load_env_file()


def transcribe_one(kind: str, opts: dict, audio: Path, language: str, out_dir: Path) -> dict:
    """Dispatch to the right client and return {text, seconds}."""
    started = time.perf_counter()

    if kind == "http":
        import json
        from multimodal_transcription_core import transcribe_http
        # the library never overwrites, so give each run its own work dir
        work = out_dir / f"_raw_{opts['model']}_{time.strftime('%Y%m%d-%H%M%S')}"
        transcribe_http(audio, opts["endpoint"], work,
                        model=opts["model"], language=language, timeout=3600)
        text = json.loads((work / "response.json").read_text())["text"]

    elif kind == "groq":
        from multimodal_transcription_core import GroqAPI, STTModel
        result = GroqAPI().speech_to_text(str(audio), model=STTModel(opts["model"]),
                                          language=language, response_format="verbose_json")
        text = result["text"]

    elif kind == "gemini":
        from multimodal_transcription_core import GeminiAPI
        result = GeminiAPI().speech_to_text(str(audio), model=opts["model"], language=language)
        text = result["text"]

    else:
        raise ValueError(f"Unknown provider kind: {kind}")

    return {"text": text, "seconds": time.perf_counter() - started}


def main() -> int:
    cfg = load_config(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "config.ini")
    load_secrets(cfg)

    audio = Path(cfg["audio"]["path"]).expanduser()
    if not audio.is_file():
        sys.exit(f"Audio not found: {audio}\nSet [audio] path in config.ini.")
    language = cfg["audio"].get("language", "ru")
    out_dir = (HERE / cfg["output"]["dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    providers = [(s.split(".", 1)[1], dict(cfg[s]))
                 for s in cfg.sections()
                 if s.startswith("provider.") and cfg[s].getboolean("enabled", False)]
    if not providers:
        sys.exit("No providers enabled in config.ini.")

    print(f"audio   : {audio.name}")
    print(f"output  : {out_dir}")
    print(f"providers: {', '.join(n for n, _ in providers)}\n")

    rows = []
    for name, opts in providers:
        print(f"-> {name} ({opts['model']}) ...", flush=True)
        try:
            r = transcribe_one(opts["kind"], opts, audio, language, out_dir)
        except Exception as exc:                      # noqa: BLE001 - report, keep going
            print(f"   FAILED: {type(exc).__name__}: {str(exc)[:160]}\n")
            rows.append((name, opts["model"], None, None, str(exc)[:40]))
            continue
        dest = out_dir / f"{audio.stem}__{opts['model']}.txt"
        dest.write_text(r["text"], encoding="utf-8")
        print(f"   {len(r['text']):,} chars in {r['seconds']:.1f}s -> {dest.name}\n")
        rows.append((name, opts["model"], len(r["text"]), r["seconds"], "ok"))

    print(f"{'provider':<10} {'model':<26} {'chars':>8} {'secs':>7}  status")
    for name, model, chars, secs, status in rows:
        c = f"{chars:,}" if chars else "-"
        s = f"{secs:.1f}" if secs else "-"
        print(f"{name:<10} {model:<26} {c:>8} {s:>7}  {status}")
    return 0 if any(r[4] == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
