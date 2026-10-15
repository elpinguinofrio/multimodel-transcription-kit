"""Choose the best available model and fall back down an explicit ranking.

Owner decision (2026-10-14): the best model is the first in the caller's ranking
that is reachable, has daily quota left and can supply every explicitly requested
feature. A failure moves to the next ranked model only when fallback is allowed
(no -m); every skip and failure is recorded in result["selection"].
"""

import os
from dataclasses import dataclass

from .errors import AudioPreparationError, NoModelAvailableError, TranscriptionError, UnsupportedFeatureError
from .rate_limiter import RateLimitError

KEY_FOR_KIND = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}
# Not model problems: another model would fail the same way.
_INPUT_ERRORS = (FileNotFoundError, IsADirectoryError, PermissionError, AudioPreparationError)


@dataclass(frozen=True)
class Availability:
    ok: bool
    reason: str = ""


def describe_error(error):
    """One line naming the failure, with the likely cause for retired models and rejected keys."""
    status = getattr(error, "status", None)
    hint = {401: "credentials rejected (expired or revoked key?)",
            403: "credentials rejected (expired or revoked key, or no access to this model?)",
            404: "model not served (retired, renamed or misspelled?)"}.get(status)
    text = f"{type(error).__name__}: {error}"
    return f"{text} - {hint}" if hint else text


def transcribe_best(audio, specs, *, probe, transcribe=None, limiter=None, allow_fallback=True, **options):
    """Transcribe with the first suitable model in ``specs`` (best first).

    ``probe(spec)`` says whether a model is reachable now (bool or Availability).
    ``options`` go to ``transcribe`` (default engine.transcribe_file), including
    ``limiter``. Raises NoModelAvailableError listing every attempt when no model
    succeeds; with allow_fallback=False the first model's own error propagates.
    """
    from .engine import resolve_features
    if transcribe is None:
        from .engine import transcribe_file as transcribe
    attempts = []
    features = {k: options.get(k) for k in ("speakers", "sounds", "timestamps")}
    for spec in specs:
        available = probe(spec)
        if not isinstance(available, Availability):
            available = Availability(bool(available), "" if available else "not reachable")
        exhausted = limiter.exhausted(spec.model) if limiter is not None and available.ok else None
        reason = available.reason if not available.ok else exhausted and f"quota: {exhausted}"
        if not reason:
            try:
                resolve_features(spec, **features)
            except UnsupportedFeatureError as error:
                reason = str(error)
        if reason:
            if not allow_fallback:
                raise NoModelAvailableError(f"{spec.name}: {reason}", attempts=[{"name": spec.name, "skipped": reason}])
            attempts.append({"name": spec.name, "skipped": reason})
            continue
        try:
            result = transcribe(audio, spec, limiter=limiter, **options)
        except _INPUT_ERRORS:
            raise
        except (TranscriptionError, RateLimitError) as error:
            if not allow_fallback:
                raise
            attempts.append({"name": spec.name, "error": describe_error(error)})
            continue
        result["selection"] = {"chosen": spec.name, "attempts": attempts, "fallback": allow_fallback}
        return result
    summary = "; ".join(f"{a['name']}: {a.get('skipped') or a.get('error')}" for a in attempts)
    raise NoModelAvailableError(f"no model could transcribe: {summary or 'none configured'}", attempts=attempts)


def _remote(url, headers, timeout):
    import requests
    try:
        return requests.get(url, headers=headers, timeout=timeout, allow_redirects=False).status_code, None
    except Exception as exc:  # a diagnostic: report the failure type, never the URL (it can hold a key)
        return None, type(exc).__name__


def verify(spec, *, timeout=10.0):
    """Ask the provider whether this key can use this model (free metadata request).

    Finds retired model names and rejected keys before a transcription fails.
    HTTP endpoints are asked for their OpenAI-style model list (``/v1/models``
    beside ``/v1/audio/transcriptions``). Never logs or returns the key.
    """
    from .config import load_env_file
    if spec.kind == "http":
        return _verify_http(spec, timeout)
    load_env_file()
    variable = spec.requires_env or KEY_FOR_KIND[spec.kind]
    key = os.environ.get(variable)
    if not key:
        return Availability(False, f"{variable} is not set")
    if spec.kind == "gemini":
        base = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
        status, failure = _remote(f"{base}/v1beta/models/{spec.model}", {"x-goog-api-key": key}, timeout)
    else:
        from .groq_client import STTAPI
        base = STTAPI.API_URL.split("/audio/")[0].rstrip("/")
        status, failure = _remote(f"{base}/models/{spec.model}", {"Authorization": f"Bearer {key}"}, timeout)
    if status == 200:
        return Availability(True, "verified: key accepted and model served")
    if status == 404:
        return Availability(False, f"model {spec.model} not served (retired, renamed or misspelled?)")
    if status in (400, 401, 403):
        return Availability(False, f"{variable} rejected (HTTP {status}: expired, revoked or no access?)")
    return Availability(True, f"unverified ({failure or f'HTTP {status}'}); will try at transcription time")


def _verify_http(spec, timeout):
    """A listed model proves the server knows the name, not which model answers a request."""
    import socket
    from urllib.parse import urlparse
    import requests
    url = urlparse(spec.endpoint)
    try:
        socket.create_connection((url.hostname, url.port or (443 if url.scheme == "https" else 80)),
                                 timeout=min(timeout, 3)).close()
    except (OSError, TypeError) as exc:
        return Availability(False, f"endpoint not answering ({type(exc).__name__})")
    head, separator, _ = spec.endpoint.partition("/audio/")
    if not separator:
        return Availability(True, "endpoint answering; model unverified (no model list beside this endpoint)")
    try:
        response = requests.get(f"{head}/models", timeout=timeout, allow_redirects=False)
    except requests.RequestException as exc:
        return Availability(True, f"endpoint answering; model unverified ({type(exc).__name__})")
    if response.status_code != 200:
        return Availability(True, f"endpoint answering; model unverified (HTTP {response.status_code})")
    try:
        listed = {item["id"]: item for item in response.json()["data"]}
    except (ValueError, KeyError, TypeError):    # requests' JSONDecodeError is a ValueError
        return Availability(True, "endpoint answering; model unverified (model list not understood)")
    if spec.model not in listed:
        return Availability(False, f"model {spec.model} not listed by the server (lists: {', '.join(listed) or 'none'})")
    if listed[spec.model].get("installed") is False:
        return Availability(False, f"model {spec.model} listed but not installed on the server")
    return Availability(True, "verified: server lists the model")
