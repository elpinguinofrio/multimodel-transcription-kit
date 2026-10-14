"""Score model transcripts against a human reference without changing either file.

Accuracy claims need a provenance record beside the reference
(reference.local.json, or --review): {"reference": <file name>, "sha256": <hex>,
"origin": "human-from-audio", "status": "reviewed", "reviewed_by": ..., "reviewed_on": ...}.
A reference seeded from a model's output, a draft, or a file whose hash changed
is refused. --allow-unreviewed scores anyway and labels the report
accuracy_claim: false.
"""

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


REVIEW_FILE = "reference.local.json"


def review_problem(reference: Path, record) -> str:
    """Why this reference cannot support an accuracy claim, or "" when it can."""
    if record is None:
        return f"no provenance record ({REVIEW_FILE} beside the reference, or --review)"
    if record.get("sha256") != hashlib.sha256(reference.read_bytes()).hexdigest():
        return "the reference changed since it was reviewed (sha256 differs)"
    if record.get("origin") != "human-from-audio":
        return f"origin is {record.get('origin')!r}; only a reference written by a person from the audio counts"
    if record.get("status") != "reviewed" or not record.get("reviewed_by"):
        return f"status is {record.get('status')!r}; it must be 'reviewed' with reviewed_by"
    return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--hypothesis", required=True, type=Path, action="append")
    parser.add_argument("--fused", type=Path,
                        help="A fused transcript, compared with the strongest individual hypothesis")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review", type=Path, help=f"Provenance record; default {REVIEW_FILE} beside the reference")
    parser.add_argument("--allow-unreviewed", action="store_true",
                        help="Score an unreviewed reference; the report makes no accuracy claim")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output file")
    review_path = args.review or args.reference.parent / REVIEW_FILE
    record = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else None
    problem = review_problem(args.reference, record)
    if problem and not args.allow_unreviewed:
        parser.error(f"Not an accuracy reference: {problem}. Use --allow-unreviewed for a labelled, "
                     "non-claim comparison.")
    reference_text = args.reference.read_text(encoding="utf-8")
    unresolved = len(re.findall(
        r"<нераз(?:борчиво|бочиво)>|\[unclear[^]]*\]",
        reference_text.lower(),
    ))
    results = []
    for hypothesis in args.hypothesis + ([args.fused] if args.fused else []):
        text = hypothesis.read_text(encoding="utf-8")
        results.append({
            "transcript": hypothesis.name,
            "transcript_sha256": hashlib.sha256(hypothesis.read_bytes()).hexdigest(),
            "fused": hypothesis == args.fused,
            **score(reference_text, text),
        })
    results.sort(key=lambda item: (item["wer_percent"], item["cer_percent"], item["transcript"]))
    fusion = None
    if args.fused:
        fused = next(r for r in results if r["fused"])
        best = next(r for r in results if not r["fused"])
        fusion = {"fused": fused["transcript"], "fused_wer_percent": fused["wer_percent"],
                  "best_individual": best["transcript"], "best_individual_wer_percent": best["wer_percent"],
                  "wer_delta_percent": round(fused["wer_percent"] - best["wer_percent"], 2),
                  "improved": fused["wer_percent"] < best["wer_percent"]}
    if problem:
        status = f"unreviewed_reference_no_accuracy_claim: {problem}"
    elif unresolved:
        status = "reviewed_human_reference_with_unresolved_spans"
    else:
        status = "reviewed_human_reference"
    report = {
        "accuracy_claim": not problem,
        "status": status,
        "review": record if not problem else None,
        "reference": args.reference.name,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "unresolved_speech_markers": unresolved,
        "normalization": "Unicode NFC; lowercase; yo-to-ye; keep alphanumeric Russian/Latin tokens; remove angle, parenthetical, square-bracket, and asterisk annotations; concatenate normalized tokens for CER",
        "limitation": "Unresolved speech spans cannot be excluded by time and may inflate or distort scores. Compare only runs of this exact sample.",
        "results": results,
        "fusion": fusion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
