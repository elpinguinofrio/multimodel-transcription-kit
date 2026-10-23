"""Chunk reconstruction: boundary speech appears once, nothing is dropped, failures surface."""

import pytest

from multimodel_transcription_kit.errors import ReconstructionError
from multimodel_transcription_kit.reconstruct import Part, reconstruct

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november".split()


def text(words):
    return " ".join(words)


def test_overlap_speech_appears_once_without_omissions():
    a = Part(1, 0, 30, text=text(WORDS[:10]))          # ... india juliet (overlap: golf..juliet)
    b = Part(2, 20, 50, text=text(WORDS[6:]))          # golf ... november
    merged = reconstruct([a, b], overlap=10)
    assert merged["text"] == text(WORDS)
    assert merged["boundaries"][0]["method"] == "overlap-match"


def test_cut_off_words_at_chunk_edges_are_replaced_by_the_complete_side():
    a = Part(1, 0, 30, text="one two three four five six seven eight nine te")
    b = Part(2, 20, 50, text="ree four five six seven eight nine ten eleven")
    merged = reconstruct([a, b], overlap=10)
    assert merged["text"] == "one two three four five six seven eight nine ten eleven"


def test_punctuation_and_case_differences_still_match():
    a = Part(1, 0, 30, text="Well, we went to the old market, and then")
    b = Part(2, 20, 50, text="went to the Old Market and then we ate.")
    merged = reconstruct([a, b], overlap=10)
    assert merged["text"] == "Well, we went to the Old Market and then we ate."


def test_three_parts_chain():
    parts = [Part(1, 0, 30, text=text(WORDS[:6])), Part(2, 20, 50, text=text(WORDS[2:10])),
             Part(3, 40, 70, text=text(WORDS[6:]))]
    assert reconstruct(parts, overlap=10)["text"] == text(WORDS)


def test_segments_are_offset_to_recording_time_and_deduplicated():
    a = Part(1, 0, 30, segments=[{"start": 0.0, "end": 12.0, "text": "alpha bravo charlie"},
                                 {"start": 21.0, "end": 29.5, "text": "delta echo foxtrot golf"}])
    b = Part(2, 20, 50, segments=[{"start": 1.0, "end": 9.5, "text": "delta echo foxtrot golf"},
                                  {"start": 12.0, "end": 20.0, "text": "hotel india"}])
    merged = reconstruct([a, b], overlap=10)
    assert merged["text"] == "alpha bravo charlie delta echo foxtrot golf hotel india"
    assert [(s["start"], s["text"]) for s in merged["segments"]] == [
        (0.0, "alpha bravo charlie"), (21.0, "delta echo foxtrot golf"), (32.0, "hotel india")]


def test_segment_providers_fall_back_to_timing_when_text_differs():
    a = Part(1, 0, 30, segments=[{"start": 0.0, "end": 10.0, "text": "first thing"},
                                 {"start": 23.0, "end": 27.0, "text": "garbled noise"}])
    b = Part(2, 20, 50, segments=[{"start": 3.0, "end": 7.0, "text": "different words"},
                                  {"start": 15.0, "end": 25.0, "text": "last thing"}])
    merged = reconstruct([a, b], overlap=10)
    # Boundary at 25 s: A's segment midpoint 25.0 goes to B, B's midpoint 25.0 stays.
    assert merged["text"] == "first thing different words last thing"
    assert merged["boundaries"][0]["method"] == "timing"


def test_text_only_without_match_uses_the_merge_model():
    seen = []

    def merge_text(tail, head):
        seen.append((tail, head))
        return "kilo lima merged"

    a = Part(1, 0, 30, text="alpha bravo kilo")
    b = Part(2, 20, 50, text="lima charlie delta")
    merged = reconstruct([a, b], overlap=10, merge_text=merge_text)
    assert seen and merged["text"].endswith("merged")
    assert merged["boundaries"][0]["method"] == "model"


def test_text_only_without_match_or_model_raises():
    a = Part(1, 0, 30, text="alpha bravo kilo")
    b = Part(2, 20, 50, text="lima charlie delta")
    with pytest.raises(ReconstructionError) as caught:
        reconstruct([a, b], overlap=10)
    assert "alpha bravo kilo" in caught.value.result["text"]
    assert "lima charlie delta" in caught.value.result["text"]


def test_failed_merge_model_raises_instead_of_duplicating():
    a = Part(1, 0, 30, text="alpha bravo kilo")
    b = Part(2, 20, 50, text="lima charlie delta")
    with pytest.raises(ReconstructionError):
        reconstruct([a, b], overlap=10, merge_text=lambda tail, head: "")


def test_merge_model_output_that_drops_text_is_rejected():
    a = Part(1, 0, 30, text=" ".join(f"a{i}" for i in range(60)))
    b = Part(2, 20, 50, text=" ".join(f"b{i}" for i in range(60)))
    with pytest.raises(ReconstructionError, match="dropped"):
        reconstruct([a, b], overlap=10, merge_text=lambda tail, head: "a1 b1")


def test_failed_and_silent_parts():
    parts = [Part(1, 0, 30, text="alpha bravo"), Part(2, 20, 50, failed=True),
             Part(3, 40, 70, no_speech=True), Part(4, 60, 90, text="charlie delta")]
    merged = reconstruct(parts, overlap=10)
    assert merged["text"] == "alpha bravo\n\n[Part 2 - TRANSCRIPTION FAILED]\n\ncharlie delta"


def test_speaker_labels_are_linked_through_the_overlap():
    a = Part(1, 0, 30, text="Speaker 1: hello there how are you\nSpeaker 2: fine thanks and you today")
    b = Part(2, 20, 50, text="Speaker 1: fine thanks and you today\nSpeaker 1: and more\nSpeaker 2: hi")
    merged = reconstruct([a, b], overlap=10, speakers=True)
    assert merged["text"].startswith("Speaker 1: hello there how are you\n"
                                     "Speaker 2: fine thanks and you today\nSpeaker 2: and more\n")
    # B's other voice could be A's Speaker 1 or someone new: never guessed.
    assert merged["text"].endswith("Speaker 3: hi")
    assert any("could not be linked" in w for w in merged["warnings"])


def test_unlinked_speakers_get_new_labels_and_a_warning():
    a = Part(1, 0, 30, text="Speaker 1: one two three four five")
    b = Part(2, 20, 50, text="Speaker 1: two three four five\nSpeaker 2: six seven")
    merged = reconstruct([a, b], overlap=10, speakers=True)
    assert merged["text"] == "Speaker 1: one two three four five\nSpeaker 2: six seven"
    a = Part(1, 0, 30, text="Speaker 1: one two three four five")
    b = Part(2, 20, 50, text="Speaker 2: two three four five\nSpeaker 1: six seven")
    merged = reconstruct([a, b], overlap=10, speakers=True)
    assert merged["text"] == "Speaker 1: one two three four five\nSpeaker 2: six seven"
    a = Part(1, 0, 30, text="Speaker 1: one two three four five")
    b = Part(2, 20, 50, text="Speaker 1: two three four five\nSpeaker 3: six seven")
    merged = reconstruct([a, b], overlap=10, speakers=True)
    assert merged["text"].endswith("Speaker 2: six seven")
    assert any("could not be linked" in w for w in merged["warnings"])
