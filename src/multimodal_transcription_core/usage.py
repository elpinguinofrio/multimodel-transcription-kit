"""Persistent request metadata, independent of quota counters and transcript files."""
import argparse
from contextlib import closing, contextmanager, nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
import uuid


_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "thought_tokens", "cached_tokens")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_requests (
    id INTEGER PRIMARY KEY, request_id TEXT NOT NULL, attempt INTEGER NOT NULL,
    started_at TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
    operation TEXT NOT NULL, status TEXT NOT NULL, http_status INTEGER,
    error_type TEXT, seconds REAL NOT NULL, response_model TEXT, finish_reason TEXT,
    input_tokens INTEGER, output_tokens INTEGER, total_tokens INTEGER,
    thought_tokens INTEGER, cached_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS usage_model_time ON usage_requests(provider, model, started_at);
"""


def _count(value):
    # Estimates, missing fields and malformed values must never become real usage.
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def usage_fields(data):
    """Normalize only provider-reported usage; never persist the response body."""
    if not isinstance(data, dict):
        data = {}
    gemini = data.get("usageMetadata")
    usage = data.get("usage")
    if isinstance(gemini, dict):
        values = [gemini.get("promptTokenCount"),
                  gemini.get("candidatesTokenCount", gemini.get("responseTokenCount")),
                  gemini.get("totalTokenCount"), gemini.get("thoughtsTokenCount"),
                  gemini.get("cachedContentTokenCount")]
    elif isinstance(usage, dict):
        values = [usage.get("prompt_tokens", usage.get("input_tokens")),
                  usage.get("completion_tokens", usage.get("output_tokens")),
                  usage.get("total_tokens"), None, None]
    else:
        values = [data.get(name) for name in _TOKEN_FIELDS]
    result = dict(zip(_TOKEN_FIELDS, map(_count, values)))
    model = data.get("modelVersion")
    result["response_model"] = model if isinstance(model, str) else None
    candidates = data.get("candidates")
    candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
    reason = candidate.get("finishReason") if isinstance(candidate, dict) else None
    result["finish_reason"] = reason if isinstance(reason, str) else None
    return result


def track(usage_log, provider, model, operation, **kwargs):
    """An explicitly disabled log performs no filesystem work."""
    return usage_log.track(provider, model, operation, **kwargs) if usage_log is not None else nullcontext({})


def _timestamp(value):
    if value is None:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Usage date filters must include a timezone, e.g. 2026-10-14T00:00:00Z")
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


class UsageLog:
    """Append-only SQLite history on local disk. The constructor creates nothing.

    Supply one instance to GeminiAPI, GroqAPI and transcribe_http. Each actual
    HTTP attempt (including retries), or Live session, becomes one row.
    Token fields are actual reported values or None, never limiter estimates.
    """
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()

    @classmethod
    def from_parser(cls, parser):
        path = parser.get("usage", "file", fallback="").strip()
        return cls(path) if path else None

    @contextmanager
    def track(self, provider, model, operation, *, request_id=None, attempt=1):
        started_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        started = time.perf_counter()
        event = {}
        error_type = None
        try:
            yield event
        except Exception as exc:
            error_type = type(exc).__name__
            if isinstance(getattr(exc, "code", None), int):
                event["http_status"] = exc.code
            raise
        finally:
            status_code = event.get("http_status")
            fields = usage_fields(event.get("response"))
            status = "error" if error_type or (status_code is not None and status_code >= 400) else "ok"
            if fields["finish_reason"] in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION"}:
                status = "blocked"
            row = dict(request_id=request_id or uuid.uuid4().hex, attempt=attempt,
                       started_at=started_at, provider=provider, model=model, operation=operation,
                       status=status, http_status=status_code, error_type=error_type,
                       seconds=time.perf_counter() - started, **fields)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # One connection per append safely supports threads and local processes.
            with closing(sqlite3.connect(self.path, timeout=30)) as db:
                db.executescript(_SCHEMA)
                columns = ", ".join(row)
                placeholders = ", ".join("?" for _ in row)
                with db:
                    db.execute(f"INSERT INTO usage_requests ({columns}) VALUES ({placeholders})", tuple(row.values()))

    def _query(self, sql, parameters):
        if not self.path.is_file():
            raise FileNotFoundError(f"No usage history at {self.path}; enable [usage] file and make a request first")
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(sql, parameters)]

    @staticmethod
    def _filters(provider=None, model=None, since=None, until=None):
        conditions, parameters = [], []
        for column, operator, value in (("provider", "=", provider), ("model", "=", model),
                                        ("started_at", ">=", _timestamp(since)),
                                        ("started_at", "<", _timestamp(until))):
            if value is not None:
                conditions.append(f"{column} {operator} ?")
                parameters.append(value)
        return (" WHERE " + " AND ".join(conditions) if conditions else ""), parameters

    def requests(self, *, provider=None, model=None, since=None, until=None, limit=100, offset=0):
        """Oldest first, paginated; use offset to fetch subsequent pages."""
        if not isinstance(limit, int) or not 1 <= limit <= 10000 or not isinstance(offset, int) or offset < 0:
            raise ValueError("limit must be 1..10000 and offset must be nonnegative")
        where, parameters = self._filters(provider, model, since, until)
        return self._query("SELECT * FROM usage_requests" + where + " ORDER BY id LIMIT ? OFFSET ?",
                           parameters + [limit, offset])

    def summary(self, *, provider=None, model=None, since=None, until=None):
        """Totals of known usage, plus explicit unknown counts, grouped by provider/model."""
        where, parameters = self._filters(provider, model, since, until)
        fields = ["provider", "model", "COUNT(*) AS requests",
                  "COUNT(DISTINCT request_id) AS calls", "SUM(attempt > 1) AS retries",
                  "SUM(status = 'ok') AS succeeded", "SUM(status != 'ok') AS failed",
                  "SUM(seconds) AS seconds"]
        for field in _TOKEN_FIELDS:
            fields.extend([f"SUM({field}) AS {field}",
                           f"SUM({field} IS NULL) AS unknown_{field.removesuffix('_tokens')}_requests"])
        return self._query("SELECT " + ", ".join(fields) + " FROM usage_requests" + where +
                           " GROUP BY provider, model ORDER BY provider, model", parameters)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fetch local usage totals or request history as JSON.")
    parser.add_argument("--db", type=Path, help="Explicit usage SQLite file")
    parser.add_argument("--config", help="Read [usage] file from a transcription config")
    parser.add_argument("--history", action="store_true", help="Individual requests instead of totals")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--since", help="Inclusive ISO timestamp with timezone")
    parser.add_argument("--until", help="Exclusive ISO timestamp with timezone")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.db:
            log = UsageLog(args.db)
        else:
            from .runner import load_config
            log = UsageLog.from_parser(load_config(args.config))
            if log is None:
                parser.error("Set [usage] file in the config, or pass --db")
        filters = dict(provider=args.provider, model=args.model, since=args.since, until=args.until)
        result = (log.requests(**filters, limit=args.limit, offset=args.offset)
                  if args.history else log.summary(**filters))
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0
