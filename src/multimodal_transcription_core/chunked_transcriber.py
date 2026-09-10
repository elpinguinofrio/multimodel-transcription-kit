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


# Re-export for convenience
split_audio = _audio_chunker.split_audio
get_duration = _audio_chunker.get_duration
cleanup_chunks = _audio_chunker.cleanup_chunks
merge_all_chunks = _chunk_merger.merge_all_chunks
GeminiAPI = _gemini_client.GeminiAPI
GeminiModel = _gemini_client.GeminiModel


def _load_logger():
    return logging


_logger_module = _load_logger()
logger = _logger_module.getLogger("chunked_transcriber")


# Threshold for chunking (in seconds) - files over this will be split
CHUNK_THRESHOLD = 1800  # 30 minutes


def _standard_transcribe(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe audio using standard (non-chunked) method.

    Args:
        audio_path: Path to audio file

    Returns:
        TranscriptionResult dict
    """
    transcriber = _load_module("transcriber")
    return transcriber.transcribe(audio_path)


def _transcribe_chunk(chunk_path: str, api: GeminiAPI) -> Dict[str, Any]:
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
        model=GeminiModel.GEMINI_3_5_FLASH,
        response_format="text"
    )
    return result


# Model for chunk transcription + merge. Must be served by the configured
# Gemini endpoint configured by GEMINI_BASE_URL.
MERGE_MODEL = "gemini-3.5-flash"


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
        import requests
        import time

        # Use configured model for merge operations
        url = self.api.API_URL.format(model=self.model)
        url += f"?key={self.api.api_key}"

        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 0.95,
            }
        }

        for attempt in range(max_retries):
            try:
                # Apply rate limiting from shared API instance
                self.api._throttle()
                self.api._rate_limit()

                res = requests.post(url, json=payload, timeout=120)

                if res.status_code != 200:
                    logger.error(f"Text generation failed (attempt {attempt+1}): {res.text}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    return ""

                data = res.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    logger.error(f"Failed to extract text from response: {data}")
                    return ""

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt+1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                return ""
            except Exception as e:
                logger.error(f"Error on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return ""

        return ""


def transcribe_large(audio_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Transcribe an audio file, handling large files via chunking.

    Drop-in replacement for standard transcriber. Auto-detects large files
    and splits them into chunks for processing.

    Args:
        audio_path: Path to the audio file

    Returns:
        TranscriptionResult dict with keys:
            - text: Transcription text
            - duration: Audio duration in seconds
            - success: Whether transcription was successful
            - chunks: Number of chunks processed (if chunked)

    Raises:
        FileNotFoundError: If the audio file does not exist
        RuntimeError: If transcription fails
    """
    path = Path(audio_path)

    # Validate file exists
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Get duration
    duration = get_duration(str(path))
    if duration is None:
        raise RuntimeError(f"Could not determine duration of {path}")

    logger.info(f"Processing {path.name} ({duration:.0f}s / {duration/60:.1f}min)")

    # If under threshold, use standard transcription
    if duration < CHUNK_THRESHOLD:
        logger.info("File under threshold, using standard transcription")
        return _standard_transcribe(str(path))

    # Large file - need to chunk
    logger.info(f"File over {CHUNK_THRESHOLD}s threshold, chunking...")

    # Split into chunks
    chunks = split_audio(str(path), chunk_sec=1800, overlap_sec=10)
    logger.info(f"Created {len(chunks)} chunks")

    # Create shared API instance for rate limiting
    api = GeminiAPI()

    # Transcribe each chunk
    transcripts: List[Dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_processing_time = 0.0

    parse_error_details: List[str] = []
    first_chunk_exception: Optional[BaseException] = None

    for i, chunk in enumerate(chunks):
        logger.info(f"Transcribing chunk {i+1}/{len(chunks)}: {chunk.start_time:.0f}s - {chunk.end_time:.0f}s")

        try:
            result = _transcribe_chunk(chunk.path, api)

            # A chunk-level [PARSE_ERROR] (e.g. Gemini PROHIBITED_CONTENT on one segment)
            # must be converted to [TRANSCRIPTION FAILED] here. Otherwise the merger
            # treats it as ordinary text, the marker survives into the merged output,
            # and pipeline-level validation discards every chunk's transcription.
            chunk_text = result.get("text", "") or ""
            if chunk_text.lstrip().startswith("[PARSE_ERROR]"):
                logger.error(f"Chunk {i+1} returned parse error: {chunk_text[:200]}")
                parse_error_details.append(chunk_text.lstrip()[len("[PARSE_ERROR]"):].strip())
                result = {**result, "text": "[TRANSCRIPTION FAILED]", "success": False}

            transcripts.append(result)

            total_input_tokens += result.get("input_tokens", 0)
            total_output_tokens += result.get("output_tokens", 0)
            total_processing_time += result.get("processing_time", 0)

            logger.info(f"Chunk {i+1} transcribed: {len(result.get('text', ''))} chars")
        except Exception as e:
            logger.error(f"Failed to transcribe chunk {i+1}: {e}")
            if first_chunk_exception is None:
                first_chunk_exception = e
            transcripts.append({"text": "[TRANSCRIPTION FAILED]", "success": False})

    # If at least one chunk failed and no chunk produced actual speech, raise so
    # the pipeline writes an error sidecar instead of producing markdown whose body
    # is only [Part N - SILENCE/FAILED] markers. All-silence is a valid output and
    # falls through; mixed speech/silence/failure also falls through (merger handles
    # markers inline). Re-raise the original chunk exception when possible so
    # pipeline.run()'s error_type heuristics (auth/timeout/conversion) still match.
    def _is_failed(t):
        return "[TRANSCRIPTION FAILED]" in (t.get("text", "") or "")

    def _is_silence(t):
        # Match the merger's silence condition (chunk_merger.py:147) plus whitespace-only
        stripped = (t.get("text", "") or "").strip()
        return not stripped or stripped.lower() == "[silence]"

    has_failed_chunk = any(_is_failed(t) for t in transcripts)
    has_speech_chunk = any(
        not _is_failed(t) and not _is_silence(t)
        for t in transcripts
    )
    if has_failed_chunk and not has_speech_chunk:
        cleanup_chunks(chunks)
        if first_chunk_exception is not None:
            raise first_chunk_exception
        if parse_error_details:
            detail = parse_error_details[0]
            raise RuntimeError(f"[PARSE_ERROR] all {len(transcripts)} chunks failed: {detail}")
        raise RuntimeError(f"All {len(transcripts)} chunks failed to transcribe")

    # Merge transcripts
    logger.info("Merging transcripts...")
    merge_api = ChunkedMergeAPI(api)
    merged_text = merge_all_chunks(transcripts, merge_api)

    # Clean up temp chunk files
    cleanup_chunks(chunks)

    # Build result
    result = {
        "text": merged_text,
        "duration": duration,
        "processing_time": total_processing_time,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "model": GeminiModel.GEMINI_3_5_FLASH.value,
        "success": True,
        "chunks": len(chunks),
    }

    logger.info(f"Transcription complete: {len(merged_text)} chars from {len(chunks)} chunks")
    return result
