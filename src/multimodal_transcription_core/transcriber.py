"""
Transcriber slice - Transcribe a single audio file to text.

This module provides a simple API to transcribe audio files using
the Gemini API. It returns the transcription text, not a file.
"""

import importlib.util
from pathlib import Path
from typing import Union

from .errors import TranscriptionError
from .rate_limiter import RateLimitError


# Supported audio file extensions
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".caf", ".mov", ".mp4", ".ogg"}


def _load_gemini_client():
    from . import gemini_client
    return gemini_client


# Load at module level for efficiency
_gemini = _load_gemini_client()
GeminiAPI = _gemini.GeminiAPI
GeminiModel = _gemini.GeminiModel


def transcribe(
    audio_path: Union[str, Path],
    diarize: bool = False,
    analyze: bool = False,
    output_format: str = "default",
    model: str = None,
    temperature: float = 1.0,
    api=None,
) -> dict:
    """
    Transcribe an audio file to text using Gemini API.

    Args:
        audio_path: Path to the audio file to transcribe.
        diarize: If True, identify and label different speakers.
        analyze: If True, extract full audio metadata (emotions, gender, background, etc.)
        output_format: Transcription style ('default', 'court', 'screenplay')
        api: Configured GeminiAPI, e.g. with a shared RateLimiter; default builds one.

    Returns:
        TranscriptionResult dict with keys:
            - text: Transcription text (or full analysis report if analyze=True)
            - model: Model name used for transcription
            - duration: Audio duration in seconds
            - processing_time: API call time in seconds
            - input_tokens: Number of input tokens
            - output_tokens: Number of output tokens
            - success: Whether transcription was successful

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the file extension is not a supported audio format.
        RuntimeError: If transcription fails due to API error.
    """
    path = Path(audio_path)

    # Validate file exists
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Validate file extension
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {path.suffix}")

    # Initialize API and transcribe
    try:
        api = api if api is not None else GeminiAPI()
        # Resolve model: param > default
        if model:
            gem_model = GeminiModel(model)
        else:
            gem_model = GeminiModel.GEMINI_3_FLASH_PREVIEW
        result = api.speech_to_text(
            audio_input=str(path),
            model=gem_model,
            response_format="text",
            diarize=diarize,
            analyze=analyze,
            output_format=output_format,
            temperature=temperature,
            # language omitted for auto-detect
        )
        return result
    except (RateLimitError, TranscriptionError):
        raise
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e
