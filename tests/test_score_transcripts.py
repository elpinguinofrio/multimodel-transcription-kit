from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_transcripts.py"
spec = spec_from_file_location("score_transcripts", SCRIPT)
module = module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("annotation", [
    "<неразборчиво>",
    "(шум)",
    "[unclear 01:35]",
    "*лай*",
])
def test_normalization_removes_each_documented_annotation(annotation):
    assert module.normalized_tokens(f"Ёж {annotation} iPhone!") == ["еж", "iphone"]


def test_identical_score_is_zero():
    result = module.score("Раз два три", "раз, два три!")
    assert result["wer_percent"] == 0
    assert result["cer_percent"] == 0


def test_edit_counts_report_substitution_deletion_and_insertion():
    result = module.score("one two three", "one too extra")
    assert result["substitutions"] == 2
    assert result["deletions"] == 0
    assert result["insertions"] == 0
    deletion = module.score("one two three", "one three")
    assert deletion["deletions"] == 1
    insertion = module.score("one three", "one two three")
    assert insertion["insertions"] == 1
