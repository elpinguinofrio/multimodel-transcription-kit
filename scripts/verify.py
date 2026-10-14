"""Public-release gate: run before every push. Exit 0 only when all checks pass.

1. The git index (what a commit would publish) equals PUBLIC_FILES.txt. Stage the
   intended files first; untracked and ignored local files are never inspected.
2. Every audio file published is an approved sample in benchmark/manifest.json,
   with the recorded sha256 and exact duration.
3. No private material: env files, personal config, references, review records,
   benchmark run folders, private folders.
4. No credentials in any published text file.
"""

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

ALLOWLIST = "PUBLIC_FILES.txt"
MANIFEST = "benchmark/manifest.json"
MEDIA = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aiff", ".caf", ".mov", ".mp4", ".webm", ".aac"}
PRIVATE = [".env", "*/.env", ".env.*", "*/.env.*", "transcription.ini", "AGENTS.md", ".private/*",
           "*human-transcription*", "*reference.local.json", "benchmark/samples/*/runs/*",
           "benchmark/samples/*/NOTES.md", "*.sqlite", ".codex/*"]
PUBLIC_EXCEPTIONS = ["*.env.example"]
CREDENTIALS = [re.compile(p) for p in (r"AIza[0-9A-Za-z_\-]{35}", r"gsk_[0-9A-Za-z]{20,}",
                                       r"sk-[A-Za-z0-9]{32,}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----")]


def allowlist(root):
    lines = (root / ALLOWLIST).read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def index(root):
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout
    return {name for name in out.decode().split("\0") if name}


def is_private(name):
    if any(fnmatch.fnmatch(name, pattern) for pattern in PUBLIC_EXCEPTIONS):
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in PRIVATE)


def wav_seconds(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def check(root):
    problems = []
    public, staged = allowlist(root), index(root)
    problems += [f"tracked but not in {ALLOWLIST}: {name}" for name in sorted(staged - public)]
    problems += [f"in {ALLOWLIST} but not staged: {name}" for name in sorted(public - staged)]
    problems += [f"private file published: {name}" for name in sorted(public | staged) if is_private(name)]

    samples = {s["path"]: s for s in json.loads((root / MANIFEST).read_text(encoding="utf-8"))["samples"]}
    for name in sorted(public):
        if Path(name).suffix.lower() in MEDIA and name not in samples:
            problems.append(f"audio not in {MANIFEST}: {name}")
    for name, sample in samples.items():
        path = root / name
        if name not in public or not path.is_file():
            problems.append(f"approved sample missing from the release: {name}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
            problems.append(f"sha256 differs from {MANIFEST}: {name}")
            continue
        if path.suffix.lower() == ".wav" and abs(wav_seconds(path) - sample["duration_seconds"]) > 1e-6:
            problems.append(f"duration differs from {MANIFEST}: {name}")

    for name in sorted(public):
        path = root / name
        if Path(name).suffix.lower() in MEDIA or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
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
