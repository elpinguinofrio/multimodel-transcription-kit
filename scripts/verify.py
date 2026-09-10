"""Verify the explicit public inventory and exact unchanged benchmark sample."""
import hashlib
import json
import subprocess
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_LINES = (ROOT / 'PUBLIC_FILES.txt').read_text().splitlines()
ALLOWED = set(ALLOWED_LINES)
BENCHMARK = ROOT / 'benchmark'

def main():
    assert len(ALLOWED) == len(ALLOWED_LINES), 'Duplicate allowlist entry'
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')) - {''}
    assert tracked == ALLOWED, f'Public inventory mismatch: {tracked ^ ALLOWED}'
    assert all((ROOT / name).is_file() for name in ALLOWED), 'Missing public file'
    public_audio = {name for name in ALLOWED if Path(name).suffix.lower() == '.wav'}
    assert public_audio == {'benchmark/sample-120s-diary/sample-120s-diary.wav'}, public_audio
    manifest = json.loads((BENCHMARK / 'manifest.json').read_text())
    audio_path = BENCHMARK / manifest['audio_path']
    assert hashlib.sha256(audio_path.read_bytes()).hexdigest() == manifest['sha256']
    with wave.open(str(audio_path), 'rb') as audio:
        assert audio.getnchannels() == manifest['channels'] == 1
        assert audio.getsampwidth() == manifest['sample_width_bytes'] == 2
        assert audio.getframerate() == manifest['sample_rate_hz'] == 48000
        assert audio.getnframes() == 5760000
        assert audio.getnframes() / audio.getframerate() == manifest['duration_seconds'] == 120
    assert manifest['human_verified_reference'] is False
    template = json.loads((BENCHMARK / 'reference.template.json').read_text())
    assert template['human_verified'] is False
    assert template['audio_sha256'] == manifest['sha256']
    print(f'PASS: exact {len(ALLOWED)}-file public allowlist; one unchanged 120-second audio file; human reference pending')

if __name__ == '__main__':
    main()
