"""
Type definitions for the transcriber module.
"""

from typing import TypedDict


class TranscriptionResult(TypedDict):
    """Result from a transcription operation."""
    text: str
    model: str
    duration: float          # Audio duration in seconds
    processing_time: float   # API call time in seconds
    input_tokens: int
    output_tokens: int
    success: bool
