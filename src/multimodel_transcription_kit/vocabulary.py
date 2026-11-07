"""Terms the speaker uses, and fixes for words models commonly mishear.

Two layers, both supplied by the caller (config [vocabulary] plus CLI flags):

- terms: sent to the model before it transcribes. Gemini gets a sentence asking
  for these spellings; Whisper-style endpoints get them as their initial prompt.
  Live models take no prompt, so hints are skipped there with a warning.
- corrections: "heard -> right" pairs applied to the finished text. A pair
  matches at the start of a word, case-insensitively, with any Russian ending
  kept ("веб-кодинга" -> "вайбкодинга"); a space or hyphen in the heard form
  also matches a space, a hyphen or nothing. Every change is recorded.
"""

import re

_SEPARATORS = re.compile(r"[\s,;]*\n[\s,;]*|;")


def parse_terms(raw):
    """Comma- or line-separated terms, in order, without blanks."""
    if not raw:
        return []
    return [t.strip() for t in re.split(r"[,\n]", raw) if t.strip()]


def parse_corrections(raw):
    """[(heard, right)] from lines or ';'-separated "heard -> right" pairs."""
    if not raw:
        return []
    pairs = []
    for item in _SEPARATORS.split(raw):
        item = item.strip()
        if not item:
            continue
        if "->" not in item:
            raise ValueError(f"correction needs 'heard -> right': {item!r}")
        heard, right = (part.strip() for part in item.split("->", 1))
        if not heard or not right:
            raise ValueError(f"correction needs 'heard -> right': {item!r}")
        pairs.append((heard, right))
    return pairs


def gemini_hint(terms):
    return ("These terms may be spoken; when they are, spell them exactly like this: "
            + ", ".join(terms) + ".") if terms else ""


def whisper_hint(terms):
    return ", ".join(terms) + "." if terms else ""


def _pattern(heard):
    parts = [re.escape(p) for p in re.split(r"[\s-]+", heard) if p]
    return re.compile(r"(?<![\w-])" + r"[\s-]?".join(parts), re.IGNORECASE)


def correct(text, corrections):
    """(corrected text, [{"heard", "right"}]) for every change made."""
    applied = []
    for heard, right in corrections or []:
        def replace(match, right=right):
            found = match.group(0)
            fixed = right[:1].upper() + right[1:] if found[:1].isupper() and right[:1].islower() else right
            applied.append({"heard": found, "right": fixed})
            return fixed
        text = _pattern(heard).sub(replace, text)
    return text, applied
