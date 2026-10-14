"""
Audio Chunker - Split large audio files into overlapping chunks using ffmpeg.

This module handles splitting audio files that are too large for single API calls.
Uses ffmpeg with stream copy (no re-encoding) for fast, lossless splitting.
"""

import importlib.util
import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _load_logger():
    return logging


_logger_module = _load_logger()
logger = _logger_module.getLogger("audio_chunker")


@dataclass
class AudioChunk:
    """Represents a chunk of audio with its metadata."""
    path: str           # Path to the chunk file
    start_time: float   # Start time in original audio (seconds)
    end_time: float     # End time in original audio (seconds)
    index: int          # Chunk index (0-based)
    temporary: bool = False  # Only files created by split_audio may be deleted


def get_duration(file_path: str) -> Optional[float]:
    """
    Get audio duration in seconds using ffprobe.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds, or None if unable to determine
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
        return None


def _run_ffmpeg(args: List[str]) -> subprocess.CompletedProcess:
    """
    Run ffmpeg command with given arguments.

    Args:
        args: List of ffmpeg arguments (excluding 'ffmpeg' itself)

    Returns:
        CompletedProcess result
    """
    cmd = ["ffmpeg"] + args
    logger.debug(f"Running ffmpeg: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result


def split_audio(
    input_path: str,
    chunk_sec: int = 1800,
    overlap_sec: int = 10,
    output_dir: Optional[str] = None
) -> List[AudioChunk]:
    """
    Split audio file into overlapping chunks using ffmpeg.

    Uses stream copy (-c copy) for fast, lossless splitting.
    Creates overlapping chunks where each chunk starts overlap_sec before
    the previous chunk ends.

    Args:
        input_path: Path to input audio file
        chunk_sec: Duration of each chunk in seconds (default: 1800 = 30 min)
        overlap_sec: Overlap duration in seconds (default: 10)
        output_dir: Directory for output chunks (default: temp directory)

    Returns:
        List of AudioChunk objects with paths and timestamps

    Example:
        For a 1-hour file with 30-min chunks and 10s overlap:
        - Chunk 0: 00:00 - 30:10 (extends 10s into the next chunk)
        - Chunk 1: 30:00 - 60:00
    """
    if chunk_sec <= 0 or overlap_sec < 0:
        raise ValueError("chunk_sec must be positive and overlap_sec nonnegative")
    # Get total duration
    total_duration = get_duration(input_path)
    if total_duration is None:
        raise RuntimeError(f"Could not determine duration of {input_path}")

    logger.info(f"Splitting {input_path} ({total_duration:.0f}s) into {chunk_sec}s chunks with {overlap_sec}s overlap")

    # If file is shorter than chunk size, return single chunk (the original file)
    if total_duration <= chunk_sec:
        logger.info("File is shorter than chunk size, returning as single chunk")
        return [AudioChunk(
            path=input_path,
            start_time=0,
            end_time=total_duration,
            index=0
        )]

    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Even chunk boundaries: n chunks start every `step` seconds and each extends
    # overlap_sec into the next, so no chunk is longer than chunk_sec + overlap_sec
    # (a provider's hard limit) and the last one is never a sliver. When the
    # duration is a multiple of chunk_sec, starts stay at round numbers.
    count = math.ceil(total_duration / chunk_sec)
    step = total_duration / count
    chunks: List[AudioChunk] = []
    input_ext = Path(input_path).suffix

    for index in range(count):
        start_time = index * step
        end_time = total_duration if index == count - 1 else min(total_duration, start_time + step + overlap_sec)
        output_path = os.path.join(output_dir, f"chunk_{index:03d}{input_ext}")
        # -ss before -i for fast input seeking; -c copy keeps the original encoding.
        _run_ffmpeg([
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(end_time - start_time),
            "-c", "copy",
            "-y",
            output_path,
        ])
        chunks.append(AudioChunk(path=output_path, start_time=start_time, end_time=end_time,
                                 index=index, temporary=True))
        logger.debug(f"Created chunk {index}: {start_time:.1f}s - {end_time:.1f}s")

    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def cleanup_chunks(chunks: List[AudioChunk]) -> None:
    """
    Delete temporary chunk files.

    Args:
        chunks: List of AudioChunk objects to clean up
    """
    for chunk in chunks:
        if chunk.temporary and os.path.exists(chunk.path):
            try:
                os.unlink(chunk.path)
                logger.debug(f"Deleted chunk: {chunk.path}")
            except OSError as e:
                logger.warning(f"Failed to delete chunk {chunk.path}: {e}")
