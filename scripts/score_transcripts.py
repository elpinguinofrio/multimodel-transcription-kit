"""Score model transcripts against a human draft without changing either file."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata


ANNOTATIONS = re.compile(r"<[^>]*>|\([^)]*\)|\[[^]]*\]|\*[^*]*\*")
TOKENS = re.compile(r"[0-9a-zа-я]+")


def normalized_tokens(text: str) -> list[str]:
    text = ANNOTATIONS.sub(" ", text)
    text = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    return TOKENS.findall(text)


def edit_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    costs = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        costs[i][0] = i
    for j in range(len(hypothesis) + 1):
        costs[0][j] = j
    for i, ref_item in enumerate(reference, 1):
        for j, hyp_item in enumerate(hypothesis, 1):
            costs[i][j] = min(
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
                costs[i - 1][j - 1] + (ref_item != hyp_item),
            )
    substitutions = deletions = insertions = 0
    i, j = len(reference), len(hypothesis)
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            substitutions += reference[i - 1] != hypothesis[j - 1]
            i -= 1
            j -= 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return {"substitutions": substitutions, "deletions": deletions, "insertions": insertions}


def score(reference_text: str, hypothesis_text: str) -> dict:
    reference = normalized_tokens(reference_text)
    hypothesis = normalized_tokens(hypothesis_text)
    if not reference:
        raise ValueError("Reference has no scoreable words")
    word_edits = edit_counts(reference, hypothesis)
    ref_chars = list("".join(reference))
    hyp_chars = list("".join(hypothesis))
    char_edits = edit_counts(ref_chars, hyp_chars)
    word_total = sum(word_edits.values())
    char_total = sum(char_edits.values())
    return {
        "reference_words": len(reference),
        "hypothesis_words": len(hypothesis),
        **word_edits,
        "word_errors": word_total,
        "wer_percent": round(100 * word_total / len(reference), 2),
        "reference_characters": len(ref_chars),
        "character_errors": char_total,
        "cer_percent": round(100 * char_total / len(ref_chars), 2),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--hypothesis", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output file")
    reference_text = args.reference.read_text(encoding="utf-8")
    unresolved = len(re.findall(
        r"<нераз(?:борчиво|бочиво)>|\[unclear[^]]*\]",
        reference_text.lower(),
    ))
    results = []
    for hypothesis in args.hypothesis:
        text = hypothesis.read_text(encoding="utf-8")
        results.append({
            "transcript": hypothesis.name,
            "transcript_sha256": hashlib.sha256(hypothesis.read_bytes()).hexdigest(),
            **score(reference_text, text),
        })
    results.sort(key=lambda item: (item["wer_percent"], item["cer_percent"], item["transcript"]))
    report = {
        "status": "provisional_owner_transcript_with_unresolved_spans" if unresolved else "owner_transcript_not_second_listener_adjudicated",
        "reference": args.reference.name,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "unresolved_speech_markers": unresolved,
        "normalization": "Unicode NFC; lowercase; yo-to-ye; keep alphanumeric Russian/Latin tokens; remove angle, parenthetical, square-bracket, and asterisk annotations; concatenate normalized tokens for CER",
        "limitation": "Unresolved speech spans cannot be excluded by time and may inflate or distort scores. Compare only runs of this exact sample.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
