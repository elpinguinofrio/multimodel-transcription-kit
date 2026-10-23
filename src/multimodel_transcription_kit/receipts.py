"""Receipts: everything needed to reproduce a transcription run, as JSON."""

import platform
import sys
import time
from dataclasses import asdict

PACKAGE = "multimodel-transcription-kit"


def client_info():
    """Which code produced a result: package version (installed metadata) and runtime."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            package_version = version(PACKAGE)
        except PackageNotFoundError:
            package_version = "source tree (not installed)"
    except ImportError:                       # pragma: no cover - Python < 3.8
        package_version = "unknown"
    return {"package": PACKAGE, "version": package_version, "python": sys.version.split()[0],
            "platform": platform.platform()}


def spec_record(spec):
    record = asdict(spec)
    if record.get("capabilities") is not None:
        record["capabilities"] = sorted(record["capabilities"])
    return record


def build_receipt(result, spec, options, **extra):
    """A JSON-ready receipt: the model spec, the options passed, input identity,
    client version, what actually ran, and (when kept) each part's raw response."""
    keys = ("text", "segments", "features", "settings", "warnings", "boundaries", "chunks", "duration",
            "processing_time", "input_tokens", "output_tokens", "no_speech", "success", "selection")
    return {
        "schema": 1,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "client": result.get("client") or client_info(),
        "spec": spec_record(spec),
        "options": options,
        "audio": result.get("audio"),
        "result": {key: result.get(key) for key in keys if key in result},
        "parts": result.get("parts"),
        **extra,
    }
