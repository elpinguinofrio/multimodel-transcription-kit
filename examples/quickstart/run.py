#!/usr/bin/env python3
"""Transcribe one recording with every provider enabled in config.ini.

Shows the intended split: this application owns configuration and secrets;
multimodal_transcription_core receives explicit values and reads no files
of its own.

    python run.py                 # uses ./config.ini
    python run.py other.ini       # uses another config
"""
import configparser
import hashlib
import json
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


def build_limiter(cfg: configparser.ConfigParser):
    """One shared queue for every provider; limits come from each [provider.*] section."""
    from multimodal_transcription_core import RateLimiter
    return RateLimiter.from_parser(cfg, section_prefix="provider.")


def transcribe_one(kind: str, opts: dict, audio: Path, language: str, out_dir: Path,
                   limiter=None) -> dict:
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
        result = GroqAPI(rate_limiter=limiter).speech_to_text(str(audio), model=STTModel(opts["model"]),
                                          language=language, response_format="verbose_json")
        text = result["text"]

    elif kind == "gemini":
        from multimodal_transcription_core import GeminiAPI
        result = GeminiAPI(rate_limiter=limiter).speech_to_text(str(audio), model=opts["model"], language=language)
        text = result["text"]

    else:
        raise ValueError(f"Unknown provider kind: {kind}")

    return {"text": text, "seconds": time.perf_counter() - started}


def main() -> int:
    cfg = load_config(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "config.ini")
    load_secrets(cfg)
    limiter = build_limiter(cfg)

    audio = Path(cfg["audio"]["path"]).expanduser()
    if not audio.is_file():
        sys.exit(f"Audio not found: {audio}\nSet [audio] path in config.ini.")
    language = cfg["audio"].get("language", "ru")

    base = Path(cfg["output"]["dir"]).expanduser()
    if not base.is_absolute():
        base = (HERE / base).resolve()

    # Every invocation gets its own directory. Nothing is ever overwritten and
    # no attempt is discarded - failures are recorded with their error text.
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = base
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "_details").mkdir(exist_ok=True)

    providers = [(s.split(".", 1)[1], dict(cfg[s]))
                 for s in cfg.sections()
                 if s.startswith("provider.") and cfg[s].getboolean("enabled", False)]
    if not providers:
        sys.exit("No providers enabled in config.ini.")

    audio_sha = hashlib.sha256(audio.read_bytes()).hexdigest()
    print(f"audio   : {audio.name}  ({audio_sha[:12]}…)")
    print(f"runs    : {run_dir}")
    print(f"providers: {', '.join(n for n, _ in providers)}\n")

    records = []
    for name, opts in providers:
        print(f"-> {name} ({opts['model']}) ...", flush=True)
        rec = {
            "run_id": run_dir.name, "provider": name, "model": opts["model"],
            "kind": opts["kind"], "endpoint": opts.get("endpoint"),
            "language": language, "audio": str(audio), "audio_sha256": audio_sha,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        started = time.perf_counter()
        try:
            r = transcribe_one(opts["kind"], opts, audio, language, run_dir, limiter=limiter)
        except Exception as exc:                  # noqa: BLE001 - record, keep going
            rec.update(status="failed", error_type=type(exc).__name__,
                       error=str(exc)[:2000], seconds=round(time.perf_counter() - started, 2))
            print(f"   FAILED: {rec['error_type']}: {rec['error'][:140]}\n")
        else:
            n = 1
            while (run_dir / f"{opts['model']} ({n}).txt").exists():
                n += 1
            dest = run_dir / f"{opts['model']} ({n}).txt"
            dest.write_text(r["text"], encoding="utf-8")
            rec["attempt"] = n
            rec.update(status="ok", chars=len(r["text"]), words=len(r["text"].split()),
                       seconds=round(r["seconds"], 2), transcript=dest.name)
            print(f"   {len(r['text']):,} chars in {r['seconds']:.1f}s -> {dest.name}\n")
        records.append(rec)

    # per-run manifest, plus one append-only line per attempt across all runs
    (run_dir / "_details" / f"run-{run_id}.json").write_text(
        json.dumps({"run_id": run_id, "audio": str(audio),
                    "audio_sha256": audio_sha, "attempts": records}, indent=2,
                   ensure_ascii=False) + "\n", encoding="utf-8")
    with (run_dir / "runs.jsonl").open("a", encoding="utf-8") as log:
        for rec in records:
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # human-readable, append-only: one line per attempt, newest at the end
    with (run_dir / "runs.log").open("a", encoding="utf-8") as log:
        for rec in records:
            out = (f"{rec['chars']:>6} chars {rec['seconds']:>6.1f}s  -> {rec['transcript']}"
                   if rec["status"] == "ok"
                   else f"{'FAILED':>12}  {rec['error_type']}: {rec['error'][:80]}")
            log.write(f"{run_id}  {rec['model']:<26} {out}\n")

    print(f"{'provider':<12} {'model':<26} {'chars':>8} {'secs':>7}  status")
    for r in records:
        c = f"{r['chars']:,}" if r.get("chars") else "-"
        sec = f"{r['seconds']:.1f}" if r.get("seconds") else "-"
        st = r["status"] if r["status"] == "ok" else f"{r['error_type']}: {r['error'][:30]}"
        print(f"{r['provider']:<12} {r['model']:<26} {c:>8} {sec:>7}  {st}")
    print(f"\nlog: {run_dir/'runs.log'}  |  json: {run_dir/'runs.jsonl'}")
    return 0 if any(r["status"] == "ok" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
