"""scripts/verify.py: the public-release gate, run on a throwaway git repository."""
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

import pytest

pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="requires git")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify.py"


def verifier():
    spec = spec_from_file_location("verify", SCRIPT)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    sample = root / "benchmark" / "samples" / "s" / "s.wav"
    sample.parent.mkdir(parents=True)
    with wave.open(str(sample), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 8000 * 3)
    (root / "benchmark" / "manifest.json").write_text(json.dumps({"samples": [{
        "path": "benchmark/samples/s/s.wav", "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
        "duration_seconds": 3.0}]}))
    (root / "README.md").write_text("hello\n")
    (root / ".gitignore").write_text("runs/\n")
    files = [".gitignore", "PUBLIC_FILES.txt", "README.md", "benchmark/manifest.json", "benchmark/samples/s/s.wav"]
    (root / "PUBLIC_FILES.txt").write_text("# published files\n" + "\n".join(files) + "\n")
    git(root, "init", "-q")
    git(root, "add", *files)
    return root


def run(root):
    return verifier().main(["--root", str(root)])


def test_a_clean_repository_passes(repo, capsys):
    assert run(repo) == 0


def test_an_ignored_local_fixture_of_a_checked_type_does_not_matter(repo):
    local = repo / "runs" / "private.wav"
    local.parent.mkdir()
    local.write_bytes(b"private recording")
    assert run(repo) == 0
    assert "runs/private.wav" not in (repo / "PUBLIC_FILES.txt").read_text()


def test_a_tracked_file_missing_from_the_allowlist_fails(repo, capsys):
    (repo / "notes.txt").write_text("x")
    git(repo, "add", "notes.txt")
    assert run(repo) == 1
    assert "notes.txt" in capsys.readouterr().out


def test_an_allowlisted_file_that_is_not_staged_fails(repo, capsys):
    (repo / "PUBLIC_FILES.txt").write_text((repo / "PUBLIC_FILES.txt").read_text() + "new.py\n")
    assert run(repo) == 1
    assert "new.py" in capsys.readouterr().out


def test_changed_benchmark_audio_fails(repo, capsys):
    (repo / "benchmark" / "samples" / "s" / "s.wav").write_bytes(b"RIFF changed")
    assert run(repo) == 1
    assert "sha256" in capsys.readouterr().out


def test_unlisted_audio_fails(repo, capsys):
    (repo / "extra.m4a").write_bytes(b"audio")
    (repo / "PUBLIC_FILES.txt").write_text((repo / "PUBLIC_FILES.txt").read_text() + "extra.m4a\n")
    git(repo, "add", "extra.m4a", "PUBLIC_FILES.txt")
    assert run(repo) == 1
    assert "extra.m4a" in capsys.readouterr().out


@pytest.mark.parametrize("name", [".env", "transcription.ini", "benchmark/samples/s/human-transcription-reference.txt",
                                  "benchmark/samples/s/reference.local.json", "benchmark/samples/s/runs/x.txt"])
def test_private_files_are_refused(repo, capsys, name):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    (repo / "PUBLIC_FILES.txt").write_text((repo / "PUBLIC_FILES.txt").read_text() + name + "\n")
    git(repo, "add", "-f", name, "PUBLIC_FILES.txt")
    assert run(repo) == 1
    assert "private" in capsys.readouterr().out


def test_credentials_in_published_text_fail(repo, capsys):
    (repo / "README.md").write_text("key = AIza" + "B" * 35 + "\n")
    git(repo, "add", "README.md")
    assert run(repo) == 1
    assert "README.md" in capsys.readouterr().out
