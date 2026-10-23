"""
Chunked Transcriber - Handle large audio files by splitting into chunks.

This module provides transcription for audio files that exceed API limits
by splitting them into overlapping chunks, transcribing each chunk,
and merging the results using LLM-based deduplication.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Load sibling modules
def _load_module(name: str):
    from importlib import import_module
    return import_module(f".{name}", __package__)


# Load dependencies
_audio_chunker = _load_module("audio_chunker")
_chunk_merger = _load_module("chunk_merger")
_gemini_client = _load_module("gemini_client")
_rate_limiter = _load_module("rate_limiter")


# Re-export for convenience
split_audio = _audio_chunker.split_audio
get_duration = _audio_chunker.get_duration
cleanup_chunks = _audio_chunker.cleanup_chunks
merge_all_chunks = _chunk_merger.merge_all_chunks
GeminiAPI = _gemini_client.GeminiAPI
GeminiModel = _gemini_client.GeminiModel
GeminiAPIError = _gemini_client.GeminiAPIError
RateLimitError = _rate_limiter.RateLimitError
from .errors import IncompleteTranscriptError
from .transcription_types import sum_known


def _load_logger():
    return logging


_logger_module = _load_logger()
logger = _logger_module.getLogger("chunked_transcriber")


# Threshold for chunking (in seconds) - files over this will be split
CHUNK_THRESHOLD = 1800  # 30 minutes


def _standard_transcribe(audio_path: str, api: Optional[GeminiAPI] = None,
                         model: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe audio using standard (non-chunked) method.

    Args:
        audio_path: Path to audio file
        api: Configured GeminiAPI to use; default builds one
        model: Gemini model name; default is transcribe()'s default

    Returns:
        TranscriptionResult dict
    """
    transcriber = _load_module("transcriber")
    return transcriber.transcribe(audio_path, api=api, model=model)


def _transcribe_chunk(chunk_path: str, api: GeminiAPI,
                      model: Union[GeminiModel, str] = GeminiModel.GEMINI_3_6_FLASH) -> Dict[str, Any]:
    """
    Transcribe a single audio chunk.

    Args:
        chunk_path: Path to chunk audio file
        api: Shared GeminiAPI instance (for rate limiting)

    Returns:
        TranscriptionResult dict
    """
    result = api.speech_to_text(
        audio_input=chunk_path,
        model=model,
        response_format="text"
    )
    return result


# Model for chunk transcription + merge. Must be served by the configured
# Gemini endpoint configured by GEMINI_BASE_URL.
MERGE_MODEL = "gemini-3.6-flash"


class ChunkedMergeAPI:
    """
    Wrapper for GeminiAPI that provides generate_text method for chunk merging.
    Uses the configured MERGE_MODEL for text generation.
    """

    def __init__(self, api: GeminiAPI, model: str = MERGE_MODEL):
        self.api = api
        self.model = model

    def generate_text(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate text using Gemini API with retry logic.

        Args:
            prompt: Text prompt for generation
            max_retries: Number of retry attempts on failure

        Returns:
            Generated text response
        """
        # Goes through GeminiAPI so merges share the rate limiter and retry policy.
        try:
            return self.api.generate_text(prompt, model=self.model, temperature=0.0,
                                          timeout=120, retries=max_retries)
        except GeminiAPIError as exc:
            logger.error(f"Merge text generation failed: {exc}")
            return ""


def transcribe_large(audio_path: Union[str, Path], api: Optional[GeminiAPI] = None,
                     model: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe a Gemini recording of any length (compatibility wrapper).

    Runs the shared path, engine.transcribe_file, with one Gemini model: files
    longer than CHUNK_THRESHOLD are split with overlap and rebuilt without
    duplicated or missing boundary speech.

    Args:
        audio_path: Path to the audio file
        api: Configured GeminiAPI, e.g. with a shared RateLimiter; default builds one
        model: One Gemini model for every request (short file, chunks, merges), so a
            limiter configured for that model covers all of them. Default keeps the
            transcribe() default for short files; gemini-3.6-flash for chunks/merges.

    Returns:
        The common result dict (text, duration, success, no_speech, chunks, usage, ...).

    Raises:
        FileNotFoundError: If the audio file does not exist
        IncompleteTranscriptError: Some parts failed; the partial result is attached
        ReconstructionError: Parts could not be joined safely
        RateLimitError subclasses: unchanged
    """
    from . import engine
    from .gemini_live import LIVE_MODELS
    model_name = model.value if isinstance(model, GeminiModel) else model
    if model_name in LIVE_MODELS:
        raise ValueError("Live models require transcribe_live; REST chunk merging is unsupported")
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if model_name is None:
        duration = engine.get_duration(str(path))
        if duration is None:
            raise RuntimeError(f"Could not determine duration of {path}")
        model_name = (GeminiModel.GEMINI_3_FLASH_PREVIEW.value if duration < CHUNK_THRESHOLD
                      else GeminiModel.GEMINI_3_6_FLASH.value)
    spec = engine.ModelSpec(name=model_name, kind="gemini", model=model_name,
                            chunk_seconds=CHUNK_THRESHOLD, merge_model=model_name if model else MERGE_MODEL)
    return engine.transcribe_file(path, spec, gemini_api=api)
