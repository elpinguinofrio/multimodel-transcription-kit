"""The quickstart example passes one shared limiter to every provider call."""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

RUN = Path(__file__).resolve().parents[1] / "examples" / "quickstart" / "run.py"


def _load_run():
    spec = importlib.util.spec_from_file_location("quickstart_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quickstart_respects_configured_daily_limit(tmp_path, monkeypatch):
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"x")
    config = tmp_path / "config.ini"
    config.write_text(f"""
[audio]
path = {audio}
language = ru

[output]
dir = {tmp_path / "out"}

[limits]
state_file = {tmp_path / "state.sqlite"}

[provider.gemini]
enabled = true
kind = gemini
model = gemini-3.5-flash-lite
rpd = 1
""")
    ok = MagicMock(status_code=200, headers={})
    ok.json.return_value = {"candidates": [{"content": {"parts": [{"text": "привет"}]}}]}
    run = _load_run()
    with patch("requests.post", return_value=ok) as post:
        for _ in range(2):
            monkeypatch.setattr(sys, "argv", ["run.py", str(config)])
            run.main()
    lines = [json.loads(l) for l in (tmp_path / "out" / "runs.jsonl").read_text().splitlines()]
    assert [l["status"] for l in lines] == ["ok", "failed"]
    assert lines[1]["error_type"] == "QuotaExhausted"
    assert post.call_count == 1
