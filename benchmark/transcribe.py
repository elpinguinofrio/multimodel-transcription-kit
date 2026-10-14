"""Benchmark runner: delegates all transcription to the installed shared core.

    python benchmark/transcribe.py --config C --audio A --runs-dir D -m NAME --label L
    python benchmark/transcribe.py --replay D/_details/receipts/<run>.json --runs-dir D

Old form (one raw HTTP request with request/response files) still works:
    python benchmark/transcribe.py --audio A --endpoint URL --output DIR
"""
import sys

from multimodal_transcription_core import bench, http_client

if __name__ == "__main__":
    if "--endpoint" in sys.argv[1:]:
        http_client.main()
    else:
        raise SystemExit(bench.main())
