"""
Type definitions for the transcriber module.
"""

import re
from typing import List, Optional, TypedDict


# Gemini prompts ask for this marker over silence; every provider's no-speech
# result is reported as no_speech=True rather than as an empty success.
NO_SPEECH_MARKER = "[SILENCE]"


class Segment(TypedDict):
    """One aligned span from a provider, in seconds from the start of the recording."""
    start: float
    end: float
    text: str


class TranscriptionResult(TypedDict, total=False):
    """Result from a transcription operation.

    Token counts are the provider's reported values, or None when unreported;
    they are never estimated or zero-filled.
    """
    text: str
    model: str
    provider: str
    duration: float          # Audio duration in seconds
    processing_time: float   # API call time in seconds
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    success: bool
    no_speech: bool          # True: the recording was transcribed and has no speech
    segments: List[Segment]  # Provider alignment, when the provider returns it


def is_no_speech(text: Optional[str]) -> bool:
    """Empty, or only the silence marker (also when each line carries a speaker label)."""
    lines = [re.sub(r"^Speaker \d+:\s*", "", line.strip()) for line in (text or "").strip().splitlines()]
    return all(not line or line.upper() == NO_SPEECH_MARKER for line in lines)


def sum_known(values) -> Optional[int]:
    """Total of reported counts; unknown if any part is unknown."""
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total
