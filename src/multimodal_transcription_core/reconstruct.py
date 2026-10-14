"""Join overlapping chunk transcripts into one transcript.

Each boundary is resolved in this order, and the method used is reported:

1. overlap-match: the longest run of identical words (case and punctuation
   ignored) in the end of one part and the start of the next. Text before the
   run comes from the earlier part, the run and everything after from the later
   part, so the overlap's speech appears once and words cut off at either edge
   are replaced by the complete side.
2. timing: providers that return aligned segments keep each segment on the side
   of the overlap's midpoint where its own midpoint falls.
3. model: text-only parts ask ``merge_text(tail, head)`` (an LLM) to merge the
   boundary windows; output that drops text is rejected.

When none applies, ReconstructionError carries the unmerged parts: a transcript
with silently duplicated or missing boundary speech is never returned.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, List, Optional

from .errors import ReconstructionError, TranscriptionError

MIN_MATCH_WORDS = 4
# Words examined on each side of a boundary: a multiple of the words expected
# in the overlap, within these bounds.
MIN_WINDOW_WORDS = 30
MAX_WINDOW_WORDS = 400
# Characters per side handed to the merge model when no words match.
MODEL_WINDOW_CHARS = 500
FAILED_MARKER = "[Part {} - TRANSCRIPTION FAILED]"

_TOKEN = re.compile(r"\S+\s*")
_LABEL = re.compile(r"(?m)^Speaker (\d+):")


@dataclass
class Part:
    """One chunk's transcript. ``start``/``end`` are seconds in the recording;
    ``segments`` (provider alignment) are relative to ``start``."""
    index: int
    start: float
    end: Optional[float]
    text: str = ""
    segments: Optional[List[dict]] = None
    failed: bool = False
    no_speech: bool = False


@dataclass
class _Run:
    """Merged text so far as tokens; each token remembers its source segment."""
    tokens: List[str] = field(default_factory=list)
    seg: List[Optional[tuple]] = field(default_factory=list)   # (start, end) absolute, or None
    has_segments: bool = False

    def text(self):
        return "".join(self.tokens).strip()


def _tokens(part):
    if part.segments is not None:
        tokens, seg = [], []
        for s in part.segments:
            span = (part.start + float(s["start"]), part.start + float(s["end"]))
            for token in _TOKEN.findall((s.get("text") or "").strip() + " "):
                tokens.append(token)
                seg.append(span)
        return _Run(tokens, seg, True)
    tokens = _TOKEN.findall(part.text.strip() + " ")
    return _Run(tokens, [None] * len(tokens), False)


def _keys(tokens, text):
    """Comparable words with their token index; speaker labels are not words."""
    labels = [(m.start(), m.end()) for m in _LABEL.finditer(text)]
    keys, offset = [], 0
    for index, token in enumerate(tokens):
        position, offset = offset, offset + len(token)
        if any(start <= position < end for start, end in labels):
            continue
        key = re.sub(r"[^\w]", "", token.lower())
        if key:
            keys.append((key, index))
    return keys


def _speaker_at(tokens, index):
    """Label of the speaker turn containing token ``index``."""
    for i in range(index, -1, -1):
        match = _LABEL.match("".join(tokens[i:i + 2]))
        if match and (i == 0 or tokens[i - 1].endswith("\n")):
            return int(match.group(1))
    return None


def _relabel(run, mapping):
    text = "".join(run.tokens)
    relabelled = _LABEL.sub(lambda m: f"Speaker {mapping[int(m.group(1))]}:", text)
    tokens = _TOKEN.findall(relabelled)
    return _Run(tokens, run.seg, run.has_segments) if len(tokens) == len(run.tokens) else run


def _window(count, overlap, duration):
    expected = count * overlap / duration if duration else count
    return int(min(MAX_WINDOW_WORDS, max(MIN_WINDOW_WORDS, 3 * expected + 20)))


def _match(left, right, overlap, left_duration, right_duration):
    left_keys = _keys(left.tokens, "".join(left.tokens))
    right_keys = _keys(right.tokens, "".join(right.tokens))
    tail = left_keys[-_window(len(left_keys), overlap, left_duration):]
    head = right_keys[:_window(len(right_keys), overlap, right_duration)]
    found = SequenceMatcher(None, [k for k, _ in tail], [k for k, _ in head],
                            autojunk=False).find_longest_match(0, len(tail), 0, len(head))
    if found.size < MIN_MATCH_WORDS:
        return None
    return [(tail[found.a + n][1], head[found.b + n][1]) for n in range(found.size)]


def _link_speakers(run, nxt, pairs, state, part):
    """Map the next part's labels onto the run's through the matched words."""
    votes = Counter()
    for left_index, right_index in pairs or []:
        a, b = _speaker_at(run.tokens, left_index), _speaker_at(nxt.tokens, right_index)
        if a is not None and b is not None:
            votes[(b, a)] += 1
    mapping = {}
    for (b, a), _ in votes.most_common():
        if b not in mapping and a not in mapping.values():
            mapping[b] = a
    labels = [int(m.group(1)) for m in _LABEL.finditer("".join(nxt.tokens))]
    unlinked = sorted({label for label in labels if label not in mapping})
    for label in unlinked:
        state["max"] += 1
        mapping[label] = state["max"]
    if unlinked:
        state["warnings"].append(
            f"speaker labels in part {part} could not be linked to earlier parts; "
            f"they were given new numbers")
    state["max"] = max([state["max"], *mapping.values()])
    return _relabel(nxt, mapping)


def _merge_model(run, nxt, merge_text, part):
    left, right = "".join(run.tokens), "".join(nxt.tokens)
    tail, head = left[-MODEL_WINDOW_CHARS:], right[:MODEL_WINDOW_CHARS]
    merged = (merge_text(tail.strip(), head.strip()) or "").strip().strip("\"'")
    if not merged:
        raise ValueError(f"merge model returned nothing at the boundary before part {part}")
    if len(merged) < 0.8 * max(len(tail.strip()), len(head.strip())) or \
            len(merged) > 1.1 * (len(tail) + len(head)):
        raise ValueError(f"merge model output at the boundary before part {part} "
                         f"dropped or invented text ({len(merged)} chars from {len(tail)}+{len(head)})")
    text = left[:-len(tail)] + merged + " " + right[len(head):]
    tokens = _TOKEN.findall(text.strip() + " ")
    return _Run(tokens, [None] * len(tokens), False)


def _join(run, nxt, overlap, durations, merge_text, speakers, state, part, boundary):
    pairs = _match(run, nxt, overlap, *durations)
    if speakers:
        nxt = _link_speakers(run, nxt, pairs, state, part)
    if pairs:
        cut_left, cut_right = pairs[0]
        return _Run(run.tokens[:cut_left] + nxt.tokens[cut_right:],
                    run.seg[:cut_left] + nxt.seg[cut_right:],
                    run.has_segments and nxt.has_segments), "overlap-match"
    if run.has_segments and nxt.has_segments:
        keep_left = [i for i, s in enumerate(run.seg) if (s[0] + s[1]) / 2 < boundary]
        keep_right = [i for i, s in enumerate(nxt.seg) if (s[0] + s[1]) / 2 >= boundary]
        return _Run([run.tokens[i] for i in keep_left] + [nxt.tokens[i] for i in keep_right],
                    [run.seg[i] for i in keep_left] + [nxt.seg[i] for i in keep_right], True), "timing"
    if merge_text is None:
        raise ValueError(f"no words match at the boundary before part {part}, the parts have "
                         f"no alignment and no merge model was given")
    return _merge_model(run, nxt, merge_text, part), "model"


def _segments(run):
    """Rebuild segments from kept tokens; truncated segments are clamped to their neighbours."""
    out = []
    for token, span in zip(run.tokens, run.seg):
        if out and out[-1]["_span"] == span:
            out[-1]["text"] += token
        else:
            out.append({"_span": span, "start": span[0], "end": span[1], "text": token})
    for previous, current in zip(out, out[1:]):
        if previous["end"] > current["start"]:
            previous["end"] = max(previous["start"], current["start"])
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in out]


def _unmerged(parts):
    pieces = [FAILED_MARKER.format(p.index) if p.failed else
              ("".join(_tokens(p).tokens).strip()) for p in parts if not p.no_speech]
    return {"text": "\n\n".join(piece for piece in pieces if piece), "success": False}


def reconstruct(parts: List[Part], *, overlap: float,
                merge_text: Optional[Callable[[str, str], str]] = None,
                speakers: bool = False) -> dict:
    """Merge chunk transcripts. Returns text, segments (absolute, or None when any
    part lacks alignment), warnings and the method used at each boundary."""
    pieces, segments, boundaries = [], [], []
    state = {"max": 0, "warnings": []}
    run, previous = None, None
    all_segments = all(p.segments is not None for p in parts if not p.failed and not p.no_speech)

    def close():
        if run is not None and run.tokens:
            pieces.append(run.text())
            if all_segments:
                segments.extend(_segments(run))

    for part in parts:
        if part.failed:
            close()
            run, previous = None, None
            pieces.append(FAILED_MARKER.format(part.index))
            continue
        if part.no_speech:
            close()
            run, previous = None, None
            continue
        nxt = _tokens(part)
        if not nxt.tokens:
            continue
        if speakers and run is None:
            labels = [int(m.group(1)) for m in _LABEL.finditer("".join(nxt.tokens))]
            if pieces and labels:
                # After a gap the labels cannot be linked to the earlier speakers.
                nxt = _link_speakers(_Run(), nxt, None, state, part.index)
            else:
                state["max"] = max([state["max"], *labels])
        if run is not None and previous.index == part.index - 1:
            boundary = (part.start + previous.end) / 2 if previous.end is not None else part.start
            durations = ((previous.end or part.start) - previous.start, (part.end or part.start) - part.start)
            try:
                run, method = _join(run, nxt, overlap, durations, merge_text, speakers, state,
                                    part.index, boundary)
            except (ValueError, TranscriptionError) as exc:   # rate limits propagate
                raise ReconstructionError(f"cannot reconstruct the transcript: {exc}",
                                          result=_unmerged(parts)) from exc
            boundaries.append({"parts": [previous.index, part.index], "method": method})
        else:
            close()
            run = nxt
        previous = part
    close()
    return {"text": "\n\n".join(pieces), "segments": segments if all_segments else None,
            "warnings": state["warnings"], "boundaries": boundaries}
