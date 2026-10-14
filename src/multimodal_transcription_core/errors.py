"""Typed failures shared by every provider and entry point.

All derive from TranscriptionError, which derives from RuntimeError so callers
that caught the old RuntimeError wrapper keep working. Rate-limit errors
(RateLimitError and subclasses) stay separate and are never wrapped.
"""


class TranscriptionError(RuntimeError):
    """Base class: transcription did not produce a complete transcript."""


class ProviderError(TranscriptionError):
    """A provider request failed (HTTP error, auth, connection, conversion).
    ``status`` is the HTTP status when there was one."""

    status = None


def with_status(error, status):
    error.status = status
    return error


class ResponseError(ProviderError):
    """The provider answered, but not with a usable complete transcript:
    blocked, malformed, truncated or a repetition loop."""

    def __init__(self, message, *, finish_reason=None, partial_text=None):
        super().__init__(message)
        self.finish_reason = finish_reason
        self.partial_text = partial_text


class IncompleteTranscriptError(TranscriptionError):
    """Some parts of a long recording failed. ``result`` holds the partial
    transcript (with failure markers, success=False); ``failed_parts`` are 1-based."""

    def __init__(self, message, *, result=None, failed_parts=()):
        super().__init__(message)
        self.result = result
        self.failed_parts = list(failed_parts)


class ReconstructionError(TranscriptionError):
    """Chunk transcripts could not be joined without risking duplicated or lost
    boundary speech. ``result`` holds the unmerged parts for inspection."""

    def __init__(self, message, *, result=None):
        super().__init__(message)
        self.result = result


class AudioPreparationError(TranscriptionError):
    """The audio could not be read, converted or split locally (e.g. ffmpeg missing)."""


class FusionError(TranscriptionError):
    """Fusing several transcripts failed; ``individual`` keeps every model's result."""

    def __init__(self, message, *, individual=(), failures=()):
        super().__init__(message)
        self.individual = list(individual)
        self.failures = list(failures)


class UnsupportedFeatureError(TranscriptionError, ValueError):
    """An explicitly requested feature cannot be supplied by the chosen model."""


class ModelUnavailableError(ProviderError):
    """The configured model is unreachable, retired, or its credentials are rejected."""


class NoModelAvailableError(TranscriptionError):
    """Every candidate model was unavailable or failed; ``attempts`` lists why."""

    def __init__(self, message, *, attempts=()):
        super().__init__(message)
        self.attempts = list(attempts)
