"""Transcript text with timestamps taken from provider alignment, never estimated."""

TIMESTAMP_MODES = ("none", "major", "phrase")
# "major": a new timestamped paragraph after a pause this long, or once a paragraph spans this long.
PARAGRAPH_PAUSE_SECONDS = 2.0
PARAGRAPH_MAX_SECONDS = 60.0


def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def render_timestamps(segments, mode):
    """``phrase``: one line per aligned segment. ``major``: paragraphs split at
    pauses and at least every minute, each starting with its time."""
    if mode == "phrase":
        return "\n".join(f"[{clock(s['start'])}] {s['text'].strip()}" for s in segments if s["text"].strip())
    paragraphs = []
    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        current = paragraphs[-1] if paragraphs else None
        if (current is None or segment["start"] - current["end"] >= PARAGRAPH_PAUSE_SECONDS
                or segment["start"] - current["start"] >= PARAGRAPH_MAX_SECONDS):
            paragraphs.append({"start": segment["start"], "end": segment["end"], "words": [text]})
        else:
            current["words"].append(text)
            current["end"] = segment["end"]
    return "\n\n".join(f"[{clock(p['start'])}] {' '.join(p['words'])}" for p in paragraphs)
