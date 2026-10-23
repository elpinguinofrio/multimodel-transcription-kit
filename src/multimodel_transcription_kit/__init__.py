"""Reusable transcription, without application storage or configuration discovery."""

from .transcriber import transcribe
from .gemini_live import transcribe_live
from .chunked_transcriber import transcribe_large
from .engine import ModelSpec, resolve_features, transcribe_file
from .selection import transcribe_best, verify as verify_model
from .fusion import transcribe_fused
from .gemini_client import GeminiAPI, GeminiAPIError, GeminiModel
from .groq_client import STTAPI as GroqAPI, STTAPIError as GroqAPIError, STTModel
from .http_client import run as transcribe_http
from .usage import UsageLog
from .errors import (
    AudioPreparationError, FusionError, IncompleteTranscriptError, ModelUnavailableError, NoModelAvailableError, ProviderError,
    ReconstructionError, ResponseError, TranscriptionError, UnsupportedFeatureError,
)
from .rate_limiter import (
    Limits, QuotaExhausted, RateLimiter, RateLimitError, RateLimitTimeout, RequestTooLarge,
)

__all__ = [
    "transcribe_file", "transcribe_best", "verify_model", "ModelSpec", "resolve_features",
    "transcribe", "transcribe_live", "transcribe_large", "transcribe_http", "GeminiAPI",
    "GeminiAPIError", "GeminiModel", "GroqAPI", "GroqAPIError", "STTModel",
    "RateLimiter", "Limits", "RateLimitError", "QuotaExhausted", "RateLimitTimeout",
    "RequestTooLarge",
    "UsageLog",
    "TranscriptionError", "ProviderError", "ResponseError", "IncompleteTranscriptError",
    "ReconstructionError", "UnsupportedFeatureError", "ModelUnavailableError",
    "NoModelAvailableError", "AudioPreparationError", "FusionError", "transcribe_fused",
]
