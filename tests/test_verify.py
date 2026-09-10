from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_public_verifier_ignores_local_benchmark_audio():
    local_audio = ROOT / "benchmark/sample-120s-diary/runs/_details/work/verifier-test.wav"
    local_audio.parent.mkdir(parents=True, exist_ok=True)
    local_audio.write_bytes(b"ignored local fixture")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/verify.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        local_audio.unlink()
    assert "one unchanged 120-second audio file" in result.stdout
