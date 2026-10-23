"""Exercise the installed CLI against a synthetic HTTP provider."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import sys
import threading


def test_installed_cli_preserves_response_and_source(tmp_path):
    captured = []
    response = {"text": "synthetic hello", "duration": 1, "words": [{"word": "hello", "start": 0, "end": 1}]}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append(self.rfile.read(int(self.headers["Content-Length"])))
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    audio = tmp_path / "synthetic.wav"
    audio.write_bytes(b"synthetic-test-audio")
    source_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
    output = tmp_path / "run"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions"
        command = [sys.executable, "-m", "multimodel_transcription_kit", "http", "--audio", str(audio),
                   "--endpoint", endpoint, "--model", "synthetic-model", "--language", "ru", "--output", str(output)]
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert json.loads((output / "response.json").read_text()) == response
        receipt = json.loads((output / "request.json").read_text())
        assert receipt["audio_sha256"] == source_hash
        assert receipt["requested_model"] == "synthetic-model"
        assert receipt["http_status"] == 200
        assert len(captured) == 1
        assert b'name="response_format"\r\n\r\nverbose_json' in captured[0]
        assert b'synthetic-test-audio' in captured[0]
        again = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert again.returncode != 0
        assert len(captured) == 1
        assert hashlib.sha256(audio.read_bytes()).hexdigest() == source_hash
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
