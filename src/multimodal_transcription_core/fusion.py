"""Run several models on one recording and combine their transcripts.

Owner decision (2026-10-14): every model's output and failure is kept; the
survivors are fused when at least two succeed; a single success is returned
labelled "not fused"; no success raises NoModelAvailableError. Whether fusion
beats the strongest single model is measured by scripts/score_transcripts.py
--fused against a reviewed human reference, never assumed.
"""

from .errors import FusionError, NoModelAvailableError, TranscriptionError
from .rate_limiter import RateLimitError
from .selection import Availability, describe_error

FUSION_MODEL = "gemini-3.6-flash"
# A fused transcript must stay within these bounds of its inputs' word counts.
MIN_RATIO, MAX_RATIO = 0.5, 1.5


def fusion_prompt(texts):
    sources = "\n\n".join(f"TRANSCRIPT {n}:\n---\n{text.strip()}\n---" for n, text in enumerate(texts, 1))
    return (
        f"Here are {len(texts)} transcripts of the same recording, each made by a different speech "
        "recognition system. Produce the single most accurate verbatim transcript. Where they "
        "disagree, keep the reading most of them share, or the one that fits the context best. "
        "Never add words that none of them contain and never drop speech that any of them "
        "transcribed confidently. Keep each language in its own script and do not translate. Keep "
        "speaker labels and *sound* descriptions only where the transcripts have them. Output only "
        f"the transcript, with no preamble.\n\n{sources}")


def gemini_fuser(model=FUSION_MODEL, *, api=None, limiter=None, usage_log=None):
    """A fuse_texts function that asks a Gemini model (through the shared limiter)."""
    def fuse(texts):
        client = api
        if client is None:
            from .gemini_client import GeminiAPI
            client = GeminiAPI(rate_limiter=limiter, usage_log=usage_log)
        return client.generate_text(fusion_prompt(texts), model=model, temperature=0.0, timeout=600)
    return fuse


def _check(fused, texts):
    words = len(fused.split())
    counts = [len(t.split()) for t in texts]
    if not fused.strip():
        return "the fusion model returned nothing"
    if words < MIN_RATIO * min(counts) or words > MAX_RATIO * max(counts):
        return (f"the fused transcript has {words} words; its inputs have {min(counts)}-{max(counts)}, "
                "so it dropped or invented text")
    return ""


def transcribe_fused(audio, specs, *, fuse_texts, transcribe=None, probe=None, limiter=None,
                     fusion_model=FUSION_MODEL, **options):
    """Transcribe ``audio`` with every spec, then fuse the successful transcripts.

    Returns {"text", "fusion": {status, inputs, failures, model}, "individual": [results]}.
    Raises FusionError (individual results attached) when the fused text is unusable.
    """
    if transcribe is None:
        from .engine import transcribe_file as transcribe
    individual, failures = [], []
    for spec in specs:
        if probe is not None:
            available = probe(spec)
            if not (available.ok if isinstance(available, Availability) else available):
                failures.append({"name": spec.name, "error": getattr(available, "reason", "") or "not reachable"})
                continue
        try:
            individual.append(transcribe(audio, spec, limiter=limiter, **options))
        except (TranscriptionError, RateLimitError) as error:
            failures.append({"name": spec.name, "error": describe_error(error)})
    if not individual:
        raise NoModelAvailableError("every fusion member failed", attempts=failures)
    names = [r.get("name") for r in individual]
    record = {"inputs": names, "failures": failures, "model": None}
    if len(individual) == 1:
        return {"text": individual[0]["text"], "fusion": {**record, "status": "not fused"},
                "individual": individual}
    texts = [r.get("plain_text", r["text"]) for r in individual]
    try:
        fused = (fuse_texts(texts) or "").strip()
    except (TranscriptionError, RateLimitError) as error:
        raise FusionError(f"fusion failed: {describe_error(error)}", individual=individual,
                          failures=failures) from error
    problem = _check(fused, texts)
    if problem:
        raise FusionError(problem, individual=individual, failures=failures)
    return {"text": fused, "fusion": {**record, "status": "fused", "model": fusion_model},
            "individual": individual}
