"""One transcription path for every provider kind and every entry point.

``transcribe_file`` is what the CLI runs and what applications call: prepare the
audio for the provider, split it within the provider's limits, send each part
through the existing client (shared limiter, retries, usage log), and rebuild
one transcript. Configuration arrives as arguments; nothing is discovered.
"""

import hashlib
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .audio_chunker import AudioChunk, get_duration, split_audio
from .audio_quality import assess as assess_audio, enhance as enhance_audio
from .chunk_merger import BOUNDARY_PROMPT
from .errors import AudioPreparationError, IncompleteTranscriptError, TranscriptionError, UnsupportedFeatureError
from .http_client import ACCEPTED_SUFFIXES as HTTP_SUFFIXES
from .rate_limiter import RateLimitError
from .receipts import client_info
from .reconstruct import Part, reconstruct
from .render import render_timestamps
from .transcription_types import NO_SPEECH_MARKER, is_no_speech, sum_known
from .usage import usage_fields

logger = logging.getLogger("engine")

KINDS = ("gemini", "groq", "http")
# Longest part sent in one request. Gemini REST: inline audio and output length;
# Live: one session stays well inside the ~10-15 minute session limit; Groq:
# its upload limit at the bitrates below; HTTP: a conservative default for local servers.
CHUNK_SECONDS = {"gemini": 1800.0, "gemini-live": 600.0, "groq": 1200.0, "http": 1800.0}
# Raw bytes per request (Gemini inline data is base64-inflated into a 20 MB request).
BYTE_LIMITS = {"gemini": 15_000_000, "groq": 24_000_000}
NATIVE_SUFFIXES = {"gemini": {".wav", ".mp3", ".m4a"},
                   "groq": {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"},
                   "http": set(HTTP_SUFFIXES)}
OVERLAP_SECONDS = 10.0
# Live sessions can be ended by the server (goAway, dropped socket) and have no
# transport retry of their own; each part gets this many fresh sessions more.
# REST clients already retry 429/5xx/connection errors, so the engine adds none.
LIVE_SESSION_RETRIES = 1
# What each kind can supply. Timestamps need provider alignment (segments), so
# Gemini, whose times would be generated text, offers none. Live returns input
# transcription only. A model section's `capabilities` overrides these.
CAPABILITIES = {"gemini": frozenset({"speakers", "sounds"}), "gemini-live": frozenset(),
                "groq": frozenset({"timestamps"}), "http": frozenset({"timestamps"})}
FEATURE_DEFAULTS = {"speakers": True, "sounds": True, "timestamps": "major"}
MERGE_MODEL = "gemini-3.6-flash"
GEMINI_DEFAULT_TEMPERATURE = 1.0


@dataclass(frozen=True)
class ModelSpec:
    """One configured model: ``name`` is the caller's label, ``model`` the provider's id."""
    name: str
    kind: str
    model: str
    endpoint: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    chunk_seconds: Optional[float] = None
    merge_model: Optional[str] = None
    requires_env: Optional[str] = None
    capabilities: Optional[frozenset] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown provider kind for model {self.name}: {self.kind!r}; use one of {KINDS}")
        if self.kind == "http" and not self.endpoint:
            raise ValueError(f"Model {self.name} is kind http and needs an endpoint")

    @classmethod
    def from_options(cls, name, options):
        """Build from a config section (strings), e.g. ``dict(config["model.<name>"])``."""
        def number(key):
            value = options.get(key)
            return float(value) if value not in (None, "") else None
        return cls(name=name, kind=options.get("kind", ""), model=options.get("model") or name,
                   endpoint=options.get("endpoint") or None, prompt=options.get("prompt") or None,
                   temperature=number("temperature"), chunk_seconds=number("chunk_seconds"),
                   merge_model=options.get("merge_model") or None,
                   requires_env=options.get("requires_env") or None,
                   capabilities=frozenset(c.strip() for c in options["capabilities"].split(",") if c.strip())
                   if options.get("capabilities") is not None else None)

    @property
    def live(self):
        from .gemini_live import LIVE_MODELS
        return self.kind == "gemini" and self.model in LIVE_MODELS

    @property
    def supports(self):
        if self.capabilities is not None:
            return self.capabilities
        return CAPABILITIES["gemini-live" if self.live else self.kind]


def resolve_features(spec, speakers=None, sounds=None, timestamps=None):
    """(applied features, warnings). None means the default (on, on, major), which
    is dropped with a warning when unsupported; an explicit request raises."""
    from .render import TIMESTAMP_MODES
    from .errors import UnsupportedFeatureError
    if timestamps is not None and timestamps not in TIMESTAMP_MODES:
        raise ValueError(f"timestamps must be one of {TIMESTAMP_MODES}, not {timestamps!r}")
    requested = {"speakers": speakers, "sounds": sounds, "timestamps": timestamps}
    reasons = {"speakers": "speaker labels", "sounds": "non-speech sound descriptions",
               "timestamps": "timestamps (they require provider alignment)"}
    applied, warnings = {}, []
    for feature, value in requested.items():
        wanted = FEATURE_DEFAULTS[feature] if value is None else value
        on = wanted not in (False, "none")
        if on and feature not in spec.supports:
            if value is not None:
                raise UnsupportedFeatureError(
                    f"{spec.name} ({spec.kind} {spec.model}) cannot supply {reasons[feature]}")
            warnings.append(f"{spec.name} cannot supply {reasons[feature]}; continuing without them")
            wanted = "none" if feature == "timestamps" else False
        applied[feature] = wanted
    return applied, warnings


class _Clients:
    """Provider clients built on first use, so an HTTP-only run needs no API keys."""

    def __init__(self, limiter, usage_log, retry_base_delay, max_retries, timeout, gemini=None, groq=None):
        self.limiter, self.usage_log = limiter, usage_log
        self.retry_base_delay, self.max_retries, self.timeout = retry_base_delay, max_retries, timeout
        self._gemini, self._groq = gemini, groq

    def _options(self):
        return dict(rate_limiter=self.limiter, usage_log=self.usage_log,
                    retry_base_delay=self.retry_base_delay, max_retries=self.max_retries)

    @property
    def gemini(self):
        if self._gemini is None:
            from .gemini_client import GeminiAPI
            self._gemini = GeminiAPI(**self._options(), **({"timeout": self.timeout} if self.timeout else {}))
        return self._gemini

    @property
    def groq(self):
        if self._groq is None:
            from .groq_client import STTAPI
            self._groq = STTAPI(**self._options(), timeout=self.timeout or 300)
        return self._groq


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _convert(source, target, args):
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-vn", *args, "-y", str(target)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise AudioPreparationError(f"ffmpeg is required to convert {source.suffix} audio for this provider") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioPreparationError(f"ffmpeg could not convert {source.name}: {exc.stderr.strip()[-300:]}") from exc
    return target


def _prepare(source, spec, work, duration):
    """The file to send: the source itself when the provider accepts it, else a converted copy."""
    if spec.live:
        return source, False                 # Live prepares 16 kHz PCM itself
    suffix = source.suffix.lower()
    if spec.kind == "http":
        if suffix in NATIVE_SUFFIXES["http"]:
            return source, False
        return _convert(source, work / "converted.wav", ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]), True
    cap = BYTE_LIMITS[spec.kind]
    dense = duration and source.stat().st_size / duration * _chunk_seconds(spec) > cap
    if suffix in NATIVE_SUFFIXES[spec.kind] and not dense:
        return source, False
    return _convert(source, work / "converted.mp3", ["-ac", "1", "-ar", "16000", "-b:a", "64k"]), True


def _chunk_seconds(spec):
    return spec.chunk_seconds or CHUNK_SECONDS["gemini-live" if spec.live else spec.kind]


def _chunk_limit(spec, path, duration):
    limit = _chunk_seconds(spec)
    cap = BYTE_LIMITS.get(spec.kind)
    if cap and duration and not spec.live:
        per_second = path.stat().st_size / duration
        if per_second * limit > cap:
            limit = cap * 0.9 / per_second
    return limit


# Word alignment is grouped into phrases: a new phrase after a pause, after
# sentence-ending punctuation, or once a phrase is this long.
PHRASE_PAUSE_SECONDS = 0.5
PHRASE_MAX_SECONDS = 10.0


def _segments(raw, words=None):
    """Provider alignment as [{start, end, text}]: segments, else words grouped into phrases."""
    out = []
    if isinstance(raw, list) and raw:
        for item in raw:
            try:
                if not isinstance(item.get("text"), str):
                    return None
                out.append({"start": float(item["start"]), "end": float(item["end"]), "text": item["text"]})
            except (AttributeError, KeyError, TypeError, ValueError):
                return None
        return out
    if not isinstance(words, list) or not words:
        return None
    for item in words:
        try:
            word, start, end = item["word"].strip(), float(item["start"]), float(item["end"])
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        last = out[-1] if out else None
        if (last is None or start - last["end"] >= PHRASE_PAUSE_SECONDS or last["text"][-1:] in ".!?…"
                or end - last["start"] > PHRASE_MAX_SECONDS):
            out.append({"start": start, "end": end, "text": word})
        else:
            last["text"] += " " + word
            last["end"] = end
    return out


def _request(spec, path, settings, clients):
    """One provider request through its existing client; the common result shape back."""
    if spec.kind == "gemini" and spec.live:
        from .gemini_live import transcribe_live
        result = transcribe_live(path, spec.model, api=clients.gemini)
        result.setdefault("request", {"model": spec.model, "transport": "live"})
        return result
    if spec.kind == "gemini":
        # A configured Gemini prompt is the whole instruction, not an addition to the built-in one.
        return clients.gemini.speech_to_text(
            str(path), model=spec.model, language=settings["language"], full_prompt=settings["prompt"],
            temperature=settings["temperature"], diarize=settings["features"]["speakers"],
            sounds=settings["features"]["sounds"], response_format="json" if settings["keep_raw"] else "text")
    if spec.kind == "groq":
        result = clients.groq.speech_to_text(str(path), model=spec.model, language=settings["language"],
                                             prompt=settings["prompt"], response_format="verbose_json")
        result["segments"] = _segments(result.get("segments"), result.get("words"))
        return result
    from .http_client import transcribe as http_transcribe
    parsed, _, _, seconds = http_transcribe(
        path, spec.endpoint, model=spec.model, language=settings["language"], prompt=settings["prompt"],
        timeout=clients.timeout or 3600, limiter=clients.limiter, usage_log=clients.usage_log,
        max_retries=clients.max_retries, base_delay=clients.retry_base_delay)
    usage = usage_fields(parsed)
    return {"text": parsed["text"], "segments": _segments(parsed.get("segments"), parsed.get("words")),
            "model": spec.model,
            "provider": "http", "processing_time": seconds, "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"], "success": True, "no_speech": is_no_speech(parsed["text"]),
            "raw_response": parsed, "request": {"endpoint": spec.endpoint, "model": spec.model,
                                                "language": settings["language"], "prompt": settings["prompt"],
                                                "response_format": "verbose_json"}}


def _merge_function(spec, clients):
    """LLM boundary merge for text-only parts: Gemini models, or an explicit merge_model."""
    model = spec.merge_model or (MERGE_MODEL if spec.kind == "gemini" else None)
    if model is None:
        return None, None

    def merge(tail, head):
        return clients.gemini.generate_text(BOUNDARY_PROMPT.format(tail=tail, head=head), model=model,
                                            temperature=0.0, timeout=120)
    return merge, model


def _request_with_retry(spec, path, settings, clients, warnings, number):
    attempts = 1 + (LIVE_SESSION_RETRIES if spec.live else 0)
    for attempt in range(1, attempts + 1):
        try:
            return _request(spec, path, settings, clients)
        except (RateLimitError, UnsupportedFeatureError, AudioPreparationError):
            raise
        except TranscriptionError as exc:
            if attempt == attempts:
                raise
            logger.warning("%s: part %d Live session failed (%s); retrying", spec.name, number, exc)
            warnings.append(f"part {number}: Live session failed ({exc}); retried in a new session")


def describe(error):
    return f"{type(error).__name__}: {error}" if error is not None else None


def _failed(result):
    text = (result.get("text") or "").lstrip()
    return text.startswith("[PARSE_ERROR]") or "[TRANSCRIPTION FAILED]" in text


def transcribe_file(audio, spec: ModelSpec, *, language=None, prompt=None, temperature=None,
                    speakers=None, sounds=None, timestamps=None, enhance=None,
                    overlap_seconds=OVERLAP_SECONDS, limiter=None, usage_log=None,
                    retry_base_delay=2.0, max_retries=4, timeout=None, gemini_api=None, groq_api=None,
                    keep_raw=False):
    """Transcribe one recording with one model; long recordings are split and rebuilt.

    speakers / sounds (bool) and timestamps ("none", "major", "phrase") default to
    None: on, on and "major" where the model can supply them, otherwise dropped
    with a warning. An explicit value the model cannot supply raises
    UnsupportedFeatureError before any request.

    Loudness and noise are assessed on every run (nothing is modified). enhance=None
    (default) writes an enhanced copy beside the source when a problem is found and
    transcribes that; True always normalises; False never enhances. The original
    is kept and result["audio"] records which file was transcribed.

    keep_raw=True adds ``parts``:
    each request's exact parameters and raw provider response (for receipts).

    Returns the common result (text, segments, usage, features, settings, audio
    identity, warnings, boundaries). Raises a TranscriptionError subclass on failure:
    IncompleteTranscriptError (partial result attached) when some parts failed,
    ReconstructionError when parts cannot be joined safely. Rate-limit errors
    propagate unchanged.
    """
    source = Path(audio)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    features, warnings = resolve_features(spec, speakers, sounds, timestamps)
    prompt = prompt if prompt is not None else spec.prompt
    if spec.kind == "gemini" and not spec.live:
        temperature = GEMINI_DEFAULT_TEMPERATURE if temperature is None and spec.temperature is None \
            else (temperature if temperature is not None else spec.temperature)
    elif temperature is not None or spec.temperature is not None:
        warnings.append(f"temperature is not sent to {spec.kind} models; the provider default applies")
        temperature = None
    if spec.live and (prompt or language):
        warnings.append("Live models take no prompt or language; both were ignored")
        prompt = language = None
    settings = {"language": language, "prompt": prompt, "temperature": temperature,
                "overlap_seconds": overlap_seconds, "features": features, "keep_raw": keep_raw}
    clients = _Clients(limiter, usage_log, retry_base_delay, max_retries, timeout, gemini_api, groq_api)
    assessment, enhanced = None, None
    try:
        assessment = assess_audio(source)
    except AudioPreparationError as exc:
        warnings.append(f"loudness not assessed: {exc}")
    if assessment is not None and enhance is not False and (assessment["problems"] or enhance):
        try:
            enhanced = enhance_audio(source, assessment, force=bool(enhance))
        except (OSError, AudioPreparationError) as exc:
            warnings.append(f"could not enhance the audio ({type(exc).__name__}: {exc}); "
                            f"transcribing the original")
    audio_path = enhanced["path"] if enhanced else source
    work = Path(tempfile.mkdtemp(prefix="transcription-core-"))
    try:
        source_duration = get_duration(str(audio_path))
        prepared, converted = _prepare(audio_path, spec, work, source_duration)
        duration = get_duration(str(prepared)) if converted else source_duration
        limit = _chunk_limit(spec, prepared, duration)
        settings["chunk_seconds"] = round(limit, 3)
        if duration is None:
            warnings.append("duration unknown (ffprobe could not read it); sent as one request without chunking")
            chunks = [AudioChunk(str(prepared), 0.0, None, 0)]
        elif duration <= limit:
            chunks = [AudioChunk(str(prepared), 0.0, duration, 0)]
        else:
            if limit <= 2 * overlap_seconds:
                raise ValueError(f"chunk length {limit:.0f}s is too short for {overlap_seconds:.0f}s overlap")
            chunks = split_audio(str(prepared), chunk_sec=limit - overlap_seconds,
                                 overlap_sec=overlap_seconds, output_dir=str(work / "chunks"))

        results, parts, errors, errors_by_part = [], [], [], {}
        for number, chunk in enumerate(chunks, start=1):
            if len(chunks) > 1:
                logger.info("%s: part %d/%d (%.0fs-%.0fs)", spec.name, number, len(chunks),
                            chunk.start_time, chunk.end_time)
            try:
                result = _request_with_retry(spec, Path(chunk.path), settings, clients, warnings, number)
                if _failed(result):
                    raise ValueError(f"part {number} returned {result['text'][:200]!r}")
            except (RateLimitError, UnsupportedFeatureError, AudioPreparationError):
                raise
            except Exception as exc:
                if len(chunks) == 1:
                    raise
                logger.error("%s: part %d failed: %s", spec.name, number, exc)
                errors.append(exc)
                errors_by_part[number] = exc
                parts.append(Part(number, chunk.start_time, chunk.end_time, failed=True))
                results.append({"input_tokens": None, "output_tokens": None})
                continue
            results.append(result)
            no_speech = result.get("no_speech", is_no_speech(result.get("text")))
            parts.append(Part(number, chunk.start_time, chunk.end_time, text=result.get("text") or "",
                              segments=result.get("segments"), no_speech=no_speech))

        speech = [p for p in parts if not p.failed and not p.no_speech]
        if errors and not speech:
            raise errors[0]
        merge_text, merge_model = _merge_function(spec, clients) if len(speech) > 1 else (None, None)
        if len(parts) == 1 and speech:
            only = results[0]
            merged = {"text": only["text"].strip(), "segments": only.get("segments"),
                      "warnings": [], "boundaries": []}
        else:
            merged = reconstruct(parts, overlap=overlap_seconds, merge_text=merge_text,
                                 speakers=features["speakers"])
            if any(b["method"] == "model" for b in merged["boundaries"]):
                settings["merge_model"] = merge_model
        warnings.extend(merged["warnings"])
        no_speech = not speech and not errors
        text = merged["text"]
        if features["timestamps"] != "none" and not no_speech:
            if merged["segments"]:
                text = render_timestamps(merged["segments"], features["timestamps"])
            elif timestamps is not None:
                raise UnsupportedFeatureError(f"{spec.name} returned no alignment, so it cannot supply "
                                              f"{timestamps} timestamps")
            else:
                warnings.append(f"{spec.name} returned no alignment; timestamps omitted")
                features["timestamps"] = "none"
        output = {
            "text": NO_SPEECH_MARKER if no_speech else text,
            "plain_text": NO_SPEECH_MARKER if no_speech else merged["text"],
            "segments": merged["segments"],
            "name": spec.name, "model": spec.model, "provider": spec.kind,
            "duration": duration,
            "processing_time": sum(r.get("processing_time") or 0 for r in results),
            "input_tokens": sum_known(r.get("input_tokens") for r in results),
            "output_tokens": sum_known(r.get("output_tokens") for r in results),
            "success": not errors, "no_speech": no_speech, "chunks": len(chunks),
            "boundaries": merged["boundaries"], "warnings": warnings, "features": features,
            "settings": {k: v for k, v in settings.items() if k not in ("features", "keep_raw")},
            "client": client_info(),
            "audio": {"source": str(source.resolve()), "sha256": _sha256(source), "assessment": assessment,
                      "enhanced": None if not enhanced else {**enhanced, "path": str(enhanced["path"])},
                      "transcribed": str(Path(audio_path).resolve()), "converted": converted},
        }
        if keep_raw:
            output["parts"] = [
                {"part": part.index, "start": part.start, "end": part.end, "failed": part.failed,
                 "request": result.get("request"), "raw_response": result.get("raw_response"),
                 "error": None if not part.failed else describe(errors_by_part.get(part.index))}
                for part, result in zip(parts, results)]
        if errors:
            failed = [p.index for p in parts if p.failed]
            raise IncompleteTranscriptError(f"{len(failed)} of {len(chunks)} parts failed: {failed}",
                                            result=output, failed_parts=failed)
        return output
    finally:
        shutil.rmtree(work, ignore_errors=True)
