"""Put one model's text onto another model's word times.

Used when the best text model cannot align words (e.g. Gemini): a timing model
(Whisper, Parakeet) is run only for its word times, and the better text is
matched onto them. Words the timing model missed get times interpolated between
their matched neighbours and are labelled so.
"""

import re
from difflib import SequenceMatcher

_STRIP = re.compile(r"[^\w]+", re.UNICODE)


def _norm(word):
    return _STRIP.sub("", word.lower().replace("ё", "е"))


def timed_words(result):
    """Absolute word times from a timing model's result (parts' raw words, else segments)."""
    words = []
    for part in result.get("parts") or []:
        if part.get("failed"):
            continue
        offset = float(part.get("start") or 0.0)
        for item in (part.get("raw_response") or {}).get("words") or []:
            try:
                words.append({"word": str(item["word"]).strip(), "start": float(item["start"]) + offset,
                              "end": float(item["end"]) + offset})
            except (KeyError, TypeError, ValueError):
                continue
    if words:
        return sorted(words, key=lambda w: (w["start"], w["end"]))
    for segment in result.get("segments") or []:       # phrase level only: spread evenly
        tokens = segment["text"].split()
        step = (float(segment["end"]) - float(segment["start"])) / max(len(tokens), 1)
        words += [{"word": t, "start": segment["start"] + i * step, "end": segment["start"] + (i + 1) * step}
                  for i, t in enumerate(tokens)]
    return words


# Not spoken words: speaker labels and *sound descriptions* the text model adds.
_NOT_SPOKEN = re.compile(r"\*[^*]*\*|<[^>]*>|\b(?:Speaker|Спикер)\s*\d+\s*:", re.IGNORECASE)
MIN_WORD_SECONDS = 0.05


def spoken_tokens(text):
    return [t for t in _NOT_SPOKEN.sub(" ", text).split() if _norm(t)]


def _fill(tokens, start, end, label):
    step = (end - start) / len(tokens)
    return [{"word": t, "start": round(start + i * step, 3), "end": round(start + (i + 1) * step, 3),
             "aligned": label} for i, t in enumerate(tokens)]


def align_words(text, timed):
    """Each spoken word of ``text`` with a start/end taken from ``timed`` ([{word,start,end}]).

    Words in both get the timing model's times ("matched"); a one-for-one difference
    takes the timed word's slot ("replaced"); the rest share the surrounding time
    ("interpolated"). The result is in order with no zero-length words.
    """
    tokens = spoken_tokens(text)
    if not tokens or not timed:
        return []
    timed = _monotonic(timed)
    matcher = SequenceMatcher(None, [_norm(t) for t in tokens], [_norm(w["word"]) for w in timed], autojunk=False)
    slots = [None] * len(tokens)            # (start, end, label) for words the timing model also heard
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                slots[i1 + k] = (timed[j1 + k]["start"], timed[j1 + k]["end"], "matched")
        elif op == "replace":
            label = "replaced" if i2 - i1 == j2 - j1 == 1 else "interpolated"
            for k, w in enumerate(_fill(tokens[i1:i2], timed[j1]["start"], timed[j2 - 1]["end"], label)):
                slots[i1 + k] = (w["start"], w["end"], w["aligned"])
    lo, hi = timed[0]["start"], timed[-1]["end"]
    out, i = [], 0
    while i < len(tokens):
        if slots[i] is not None:
            out.append({"word": tokens[i], "start": slots[i][0], "end": slots[i][1], "aligned": slots[i][2]})
            i += 1
            continue
        j = i
        while j < len(tokens) and slots[j] is None:
            j += 1
        run = tokens[i:j]
        start = out[-1]["end"] if out else lo
        end = slots[j][0] if j < len(tokens) else hi
        if end - start >= MIN_WORD_SECONDS * len(run):
            out += _fill(run, start, end, "interpolated")
        elif out:                            # no room: share the previous word's time
            prev = out.pop()
            shared = _fill([prev["word"]] + run, prev["start"], end, "interpolated")
            shared[0]["aligned"] = prev["aligned"]
            out += shared
        elif j < len(tokens):                # leading words: share the next word's time
            nxt = slots[j]
            shared = _fill(run + [tokens[j]], min(start, nxt[0]), nxt[1], "interpolated")
            shared[-1]["aligned"] = nxt[2]
            out += shared
            j += 1
        else:
            out += _fill(run, lo, max(hi, lo + MIN_WORD_SECONDS * len(run)), "interpolated")
        i = j
    return _repair(out)


def _monotonic(timed):
    """Timing words sorted, non-overlapping and never zero-length (Whisper's can overlap)."""
    out, last = [], None
    for w in sorted(timed, key=lambda w: (w["start"], w["end"])):
        start = max(w["start"], last) if last is not None else w["start"]
        end = max(w["end"], start + MIN_WORD_SECONDS)
        out.append({**w, "start": round(start, 3), "end": round(end, 3)})
        last = end
    return out


def _repair(words):
    """Last guard: every word starts at or after the previous end and lasts > 0."""
    last = None
    for w in words:
        if last is not None and w["start"] < last:
            w["start"] = last
        if w["end"] <= w["start"]:
            w["end"] = round(w["start"] + MIN_WORD_SECONDS, 3)
        last = w["end"]
    return words
