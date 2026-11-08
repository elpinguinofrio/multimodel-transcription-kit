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


def transcribe_best(audio, specs, *, probe, transcribe=None, limiter=None, allow_fallback=True,
                    timing_specs=None, **options):
    """Transcribe with the first suitable model in ``specs`` (best first).

    ``probe(spec)`` says whether a model is reachable now (bool or Availability).
    ``options`` go to ``transcribe`` (default engine.transcribe_file), including
    ``limiter``. Raises NoModelAvailableError listing every attempt when no model
    succeeds; with allow_fallback=False the first model's own error propagates.
    ``timing_specs`` (best first) supply word times when the chosen text model cannot;
    default: the timestamp-capable models of ``specs`` in their order.
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
        exhausted = limiter.exhausted(spec.limit_key) if limiter is not None and available.ok else None
        reason = available.reason if not available.ok else exhausted and f"quota: {exhausted}"
        borrow_times = False
        if not reason:
            try:
                resolve_features(spec, **features)
            except UnsupportedFeatureError as error:
                reason = str(error)
                # Owner decision (2026-11-07): word times never demote the text to a weaker
                # model. The best model writes the text; a timing model supplies only times.
                if features["timestamps"] not in (None, "none") and "timestamps" not in spec.supports:
                    try:
                        resolve_features(spec, **{**features, "timestamps": "none"})
                        reason, borrow_times = "", True
                    except UnsupportedFeatureError:
                        pass
        if reason:
            if not allow_fallback:
                raise NoModelAvailableError(f"{spec.name}: {reason}", attempts=[{"name": spec.name, "skipped": reason}])
            attempts.append({"name": spec.name, "skipped": reason})
            continue
        try:
            if borrow_times:
                result = transcribe(audio, spec, limiter=limiter, **{**options, "timestamps": "none"})
                result = _add_times(audio, result, spec, specs if timing_specs is None else timing_specs,
                                    probe, transcribe, limiter, options)
            elif features["timestamps"] not in (None, "none"):
                # Native word times: expose them as result["words"] too, so callers never
                # need to know which model ran or where its raw response keeps them.
                result = transcribe(audio, spec, limiter=limiter, **{**options, "keep_raw": True})
                if "words" not in result:
                    from .alignment import timed_words
                    result["words"] = timed_words({"parts": result.get("parts")})
            else:
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


def _add_times(audio, result, text_spec, specs, probe, transcribe, limiter, options):
    """Align ``result``'s text onto word times from the best timing-capable model."""
    from .alignment import align_words, timed_words
    from .engine import _segments
    from .render import render_timestamps
    timing_specs = [s for s in specs if s is not text_spec and "timestamps" in s.supports]
    try:
        timing = transcribe_best(audio, timing_specs, probe=probe, transcribe=transcribe, limiter=limiter,
                                 allow_fallback=True, **{**options, "keep_raw": True})
    except NoModelAvailableError as error:
        raise NoModelAvailableError(f"{text_spec.name} cannot align words and no timing model could: {error}",
                                    attempts=error.attempts) from error
    if result.get("no_speech"):
        words = []
    else:
        words = align_words(result.get("plain_text", result["text"]), timed_words(timing))
        if not words:
            raise UnsupportedFeatureError(f"timing model {timing.get('name')} returned no word times")
        segments = _segments(None, words)
        result["segments"] = segments
        result["text"] = render_timestamps(segments, options["timestamps"])
    result["words"] = words
    result["features"] = {**(result.get("features") or {}), "timestamps": options["timestamps"]}
    result["alignment"] = {
        "text_model": text_spec.name, "timing_model": timing.get("name"),
        "matched": sum(w["aligned"] == "matched" for w in words), "words": len(words),
        "timing_selection": timing.get("selection"), "timing_text": timing.get("plain_text", timing.get("text")),
        "timing_parts": timing.get("parts")}
    return result


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
    if spec.kind == "agy":
        from .antigravity_client import binary
        return (Availability(True, "agy CLI installed; quota not checked (agy reports none up front)")
                if binary() else Availability(False, "the agy CLI is not installed"))
    load_env_file()
    variable = spec.requires_env or KEY_FOR_KIND.get(spec.kind)
    key = os.environ.get(variable) if variable else None
    if variable and not key:
        return Availability(False, f"{variable} is not set")
    if spec.kind == "openai":
        # The model list works on Groq, OpenAI and local servers alike (/models/<id> does not).
        base = (spec.base_url or "https://api.openai.com/v1").rstrip("/")
        return _verify_http(spec, timeout, endpoint=f"{base}/audio/transcriptions",
                            headers={"Authorization": f"Bearer {key}"} if key else None)
    if spec.kind == "gemini":
        base = (spec.base_url or os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com").rstrip("/")
        status, failure = _remote(f"{base}/v1beta/models/{spec.model}", {"x-goog-api-key": key}, timeout)
    else:
        from .groq_client import STTAPI
        base = (spec.base_url or STTAPI.API_URL.split("/audio/")[0]).rstrip("/")
        status, failure = _remote(f"{base}/models/{spec.model}",
                                  {"Authorization": f"Bearer {key}"} if key else {}, timeout)
    if status == 200:
        return Availability(True, "verified: key accepted and model served")
    if status == 404:
        return Availability(False, f"model {spec.model} not served (retired, renamed or misspelled?)")
    if status in (400, 401, 403):
        return Availability(False, f"{variable} rejected (HTTP {status}: expired, revoked or no access?)")
    return Availability(True, f"unverified ({failure or f'HTTP {status}'}); will try at transcription time")


def _verify_http(spec, timeout, endpoint=None, headers=None):
    """A listed model proves the server knows the name, not which model answers a request."""
    import socket
    from urllib.parse import urlparse
    import requests
    endpoint = endpoint or spec.endpoint
    url = urlparse(endpoint)
    try:
        socket.create_connection((url.hostname, url.port or (443 if url.scheme == "https" else 80)),
                                 timeout=min(timeout, 3)).close()
    except (OSError, TypeError) as exc:
        return Availability(False, f"endpoint not answering ({type(exc).__name__})")
    head, separator, _ = endpoint.partition("/audio/")
    if not separator:
        return Availability(True, "endpoint answering; model unverified (no model list beside this endpoint)")
    try:
        response = requests.get(f"{head}/models", headers=headers, timeout=timeout, allow_redirects=False)
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
