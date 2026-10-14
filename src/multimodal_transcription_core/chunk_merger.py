"""
Chunk Merger - LLM-based deduplication of overlapping transcript chunks.

Uses Gemini to intelligently merge the overlapping regions between
consecutive transcript chunks, removing duplicated text.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List


def _load_logger():
    return logging


_logger_module = _load_logger()
logger = _logger_module.getLogger("chunk_merger")


# Window size for boundary extraction (chars)
# ~500 chars ≈ 30 seconds of speech, plenty for 10s overlap
WINDOW = 500


def merge_adjacent_chunks(chunk_a: str, chunk_b: str, api: Any) -> str:
    """
    Merge two overlapping transcripts using LLM.

    Strategy:
    1. Extract last WINDOW chars of chunk A (tail)
    2. Extract first WINDOW chars of chunk B (head)
    3. LLM merges ONLY this boundary region
    4. Stitch: chunk_a[:-WINDOW] + merged_boundary + chunk_b[WINDOW:]

    Args:
        chunk_a: Text of the first chunk (ends with overlap)
        chunk_b: Text of the second chunk (starts with overlap)
        api: Gemini API client with generate_text method

    Returns:
        Merged text with overlap deduplicated
    """
    # Handle short chunks - use full text if shorter than window
    len_a = len(chunk_a)
    len_b = len(chunk_b)

    if len_a <= WINDOW and len_b <= WINDOW:
        # Both chunks are short - merge entirely
        tail_a = chunk_a
        head_b = chunk_b
        prefix_a = ""
        suffix_b = ""
    elif len_a <= WINDOW:
        # Only A is short
        tail_a = chunk_a
        head_b = chunk_b[:WINDOW]
        prefix_a = ""
        suffix_b = chunk_b[WINDOW:]
    elif len_b <= WINDOW:
        # Only B is short
        tail_a = chunk_a[-WINDOW:]
        head_b = chunk_b
        prefix_a = chunk_a[:-WINDOW]
        suffix_b = ""
    else:
        # Normal case - both longer than window
        tail_a = chunk_a[-WINDOW:]
        head_b = chunk_b[:WINDOW]
        prefix_a = chunk_a[:-WINDOW]
        suffix_b = chunk_b[WINDOW:]

    # Build prompt for LLM
    prompt = f"""These two text segments are from consecutive audio transcription chunks.
They overlap by approximately 10 seconds of audio. Combine them into ONE coherent text,
removing any duplicated words/phrases from the overlap.

END OF CHUNK A:
---
{tail_a}
---

START OF CHUNK B:
---
{head_b}
---

Output ONLY the combined text (the merged boundary region). No explanation, no preamble."""

    logger.debug(f"Merging chunks: A ends with '{tail_a[-50:]}...', B starts with '...{head_b[:50]}'")

    # Call LLM
    merged_boundary = api.generate_text(prompt)

    # Clean up LLM response (remove any surrounding quotes or whitespace)
    merged_boundary = merged_boundary.strip().strip('"\'')

    # A failed merge must not drop words: keep both sides, overlap may repeat.
    if not merged_boundary:
        logger.warning("Boundary merge returned nothing; keeping both sides unmerged")
        merged_boundary = f"{tail_a} {head_b}"

    # Stitch: prefix + merged_boundary + suffix
    result = prefix_a + merged_boundary + suffix_b

    logger.debug(f"Merged result length: {len(result)} chars")
    return result


def merge_all_chunks(transcripts: List[Dict], api: Any) -> str:
    """
    Merge all transcript chunks into a single coherent text.

    Processes pairs sequentially, using LLM to merge each boundary.
    Adds [Part N] markers between chunks for context.

    Strategy: Merge all text first WITHOUT markers, then join with markers
    at the end. This prevents markers from being in the LLM merge window.

    Args:
        transcripts: List of dicts with 'text' key containing chunk transcripts
        api: Gemini API client with generate_text method

    Returns:
        Single merged transcript with part markers
    """
    if not transcripts:
        return ""

    if len(transcripts) == 1:
        return transcripts[0]["text"]

    logger.info(f"Merging {len(transcripts)} transcript chunks")

    # Collect merged segments and their part numbers
    # Strategy: merge without markers, then join with markers at the end
    merged_segments = []  # List of (merged_text, part_numbers_list)

    current_text = None
    current_parts = []

    for i, t in enumerate(transcripts):
        chunk_text = t["text"]
        part_num = i + 1
        is_silence = not chunk_text or chunk_text.strip() == "[SILENCE]"
        is_failed = chunk_text and "[TRANSCRIPTION FAILED]" in chunk_text

        if is_silence:
            # Save current segment and start fresh
            if current_text is not None:
                merged_segments.append((current_text, current_parts))
                current_text = None
                current_parts = []
            # Add silence marker as its own segment
            merged_segments.append((f"[Part {part_num} - SILENCE]", [part_num]))
        elif is_failed:
            # Don't try to merge failed chunks - treat like silence
            if current_text is not None:
                merged_segments.append((current_text, current_parts))
                current_text = None
                current_parts = []
            merged_segments.append((f"[Part {part_num} - TRANSCRIPTION FAILED]", [part_num]))
        else:
            if current_text is None:
                # Start new segment
                current_text = chunk_text
                current_parts = [part_num]
            else:
                # Merge with current (no markers in the merge!)
                current_text = merge_adjacent_chunks(current_text, chunk_text, api)
                current_parts.append(part_num)

    # Don't forget the last segment
    if current_text is not None:
        merged_segments.append((current_text, current_parts))

    # Now join segments with markers
    result_parts = []
    for i, (text, parts) in enumerate(merged_segments):
        if i > 0:
            result_parts.append("\n\n")
        # Add part marker header if multiple parts were merged
        if len(parts) > 1:
            result_parts.append(f"[Parts {parts[0]}-{parts[-1]}]\n\n")
        elif len(parts) == 1 and i > 0:
            result_parts.append(f"[Part {parts[0]}]\n\n")
        result_parts.append(text)

    result = "".join(result_parts)
    logger.info(f"Merged transcript: {len(result)} chars total")
    return result
