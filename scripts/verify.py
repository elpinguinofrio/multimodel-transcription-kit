"""Public-release gate: run before every push. Exit 0 only when all checks pass.

1. Inspect the git index (what a commit would publish), including staged bytes.
   Untracked files and unstaged working-copy edits are never inspected.
2. Every audio file published is an approved sample in benchmark/manifest.json,
   with the recorded sha256 and exact duration.
3. No private material: env files, personal config, references, review records,
   benchmark run folders, private folders.
4. No credentials in any published text file.
"""

import argparse
import fnmatch
import hashlib
import io
import json
import re
import subprocess
import wave
from pathlib import Path

MANIFEST = "benchmark/manifest.json"
MEDIA = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aiff", ".caf", ".mov", ".mp4", ".webm", ".aac"}
PRIVATE = [".env", "*/.env", ".env.*", "*/.env.*", "transcription.ini", "AGENTS.md", ".private/*",
           "*human-transcription*", "*reference.local.json", "benchmark/samples/*/runs/*",
           "benchmark/samples/*/NOTES.md", "*.sqlite", ".codex/*"]
PUBLIC_EXCEPTIONS = ["*.env.example"]
CREDENTIALS = [re.compile(p) for p in (r"AIza[0-9A-Za-z_\-]{35}", r"gsk_[0-9A-Za-z]{20,}",
                                       r"sk-[A-Za-z0-9]{32,}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----")]


def index(root):
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout
    return {name: subprocess.run(["git", "-C", str(root), "show", f":{name}"],
                                check=True, capture_output=True).stdout
            for name in out.decode().split("\0") if name}


def is_private(name):
    if any(fnmatch.fnmatch(name, pattern) for pattern in PUBLIC_EXCEPTIONS):
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in PRIVATE)


def wav_seconds(data):
    with wave.open(io.BytesIO(data), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def check(root):
    problems = []
    public = index(root)
    problems += [f"private file published: {name}" for name in sorted(public) if is_private(name)]
    try:
        samples = {s["path"]: s for s in json.loads(public[MANIFEST])["samples"]}
    except (KeyError, ValueError, TypeError) as exc:
        return problems + [f"missing or invalid staged {MANIFEST}: {exc}"]
    for name in sorted(public):
        if Path(name).suffix.lower() in MEDIA and name not in samples:
            problems.append(f"audio not in {MANIFEST}: {name}")
    for name, sample in samples.items():
        if name not in public:
            problems.append(f"approved sample missing from the release: {name}")
            continue
        if hashlib.sha256(public[name]).hexdigest() != sample["sha256"]:
            problems.append(f"sha256 differs from {MANIFEST}: {name}")
            continue
        if Path(name).suffix.lower() == ".wav" and abs(wav_seconds(public[name]) - sample["duration_seconds"]) > 1e-6:
            problems.append(f"duration differs from {MANIFEST}: {name}")

    for name in sorted(public):
        if Path(name).suffix.lower() in MEDIA:
            continue
        try:
            text = public[name].decode("utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in CREDENTIALS):
            problems.append(f"credential-like string in {name}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    problems = check(args.root.resolve())
    for problem in problems:
        print(f"FAIL {problem}")
    print("verify: ok" if not problems else f"verify: {len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
