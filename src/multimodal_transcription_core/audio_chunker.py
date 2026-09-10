"""
Audio Chunker - Split large audio files into overlapping chunks using ffmpeg.

This module handles splitting audio files that are too large for single API calls.
Uses ffmpeg with stream copy (no re-encoding) for fast, lossless splitting.
"""

import importlib.util
import logging
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
        - Chunk 0: 00:00 - 30:00
        - Chunk 1: 29:50 - 60:00 (starts 10s before previous ends)
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

    # Calculate chunk boundaries with overlap
    # Overlap is achieved by EXTENDING each chunk into the next one's territory
    # This keeps start times at round numbers (0, 10min, 20min, etc.)
    chunks: List[AudioChunk] = []
    step = chunk_sec  # Chunks start at round intervals (0, 600, 1200, ...)
    start_time = 0.0
    index = 0

    input_ext = Path(input_path).suffix

    while start_time < total_duration:
        # Calculate end time - extend by overlap_sec to create overlap with next chunk
        # If remaining audio is less than 1.5x chunk size, extend to end
        remaining = total_duration - start_time
        if remaining <= chunk_sec * 1.5:
            end_time = total_duration
        else:
            end_time = start_time + chunk_sec + overlap_sec  # Extend into next chunk

        # Generate output path
        output_path = os.path.join(output_dir, f"chunk_{index:03d}{input_ext}")

        # ffmpeg command to extract chunk
        # -ss before -i for fast seek (input seeking)
        # -t for duration
        # -c copy for stream copy (no re-encoding)
        args = [
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(end_time - start_time),
            "-c", "copy",
            "-y",  # Overwrite output
            output_path
        ]

        _run_ffmpeg(args)

        chunks.append(AudioChunk(
            path=output_path,
            start_time=start_time,
            end_time=end_time,
            index=index,
            temporary=True,
        ))

        logger.debug(f"Created chunk {index}: {start_time:.1f}s - {end_time:.1f}s")

        # If we've reached the end, stop
        if end_time >= total_duration:
            break

        # Move to next chunk
        start_time += step
        index += 1

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
