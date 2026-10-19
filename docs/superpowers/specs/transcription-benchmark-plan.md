> Historical design/implementation record. For current setup and structure, see
> [README](../../../README.md) and the [current roadmap](../../roadmap.md).

# Plan: transcription benchmark for my diaries

## Goal
Pick the setup (model + route + prompt) that gives the best transcripts of my voice
diaries within my daily limits, then learn which levers improve it.

## Rules
- A **setup** is frozen: model, route, exact prompt bytes, settings, client version. Stored in a receipt.
- Ranking compares whole setups on the same clips. Lever tests (prompt, temperature, audio prep)
  change one thing on one setup.
- Every setup runs the same number of times on every clip. Order of API calls is interleaved.
- Raw provider output and the client's cleaned output are both kept and scored.
- The existing 120 s sample (C1) is diagnostic only: its reference was edited from a Gemini 3.8 run.

## Phase 1 — provisional decision (no new code beyond the scorer)
Clips (private, `.private/benchmark/`, never committed), 60–90 s each, references written by me
from audio alone without seeing any model output:

| Clip | Content |
|---|---|
| A | quiet, Russian |
| B | outdoor noise, Russian |
| C | Russian with English words |
| D | fast / quiet / mumbled |
| E | Spanish (with Russian/English mixed in if natural) |

Setups, 1 run per clip:

| Setup | Daily limit | Role |
|---|---|---|
| Gemini 3.5 Flash Lite + prompt P1 (REST) | 500 calls | bulk candidate |
| Groq Whisper large-v3-turbo | 2,000 calls / 28,800 audio s (confirm on account page) | bulk candidate |
| Parakeet TDT 0.6B v3 (host) | none | bulk candidate |
| Gemini 3.8 Flash + prompt P1 (REST) | 20 calls | quality ceiling only |

Scoring (reuse, don't invent): `jiwer.process_words`; Whisper's multilingual `BasicTextNormalizer`
plus a short tested list of Russian mappings; pooled substitutions/deletions/insertions over all clips
as the main number, per-clip table beside it; name precision/recall on a per-clip glossary.

Result: one table, one row per setup. If the top two are close or win on different clips,
grow to 12–20 clips from different days before anything else.

## Phase 2 — unlimited-call Live models (needs a Live client in the core)
Each its own row, same clips, same prompt as the Phase 1 Gemini winner:
Gemini 3.5 Transcribe Live (20K tokens/min), Gemini 3 Flash Live, Gemini 3.8 Live (65K tokens/min).
Also measured: seconds per audio minute (Live streams at about real-time speed) and session breaks
on a 30–60 min diary.

## Phase 3 — improve the winner (one lever at a time)
| Lever | Variants |
|---|---|
| Prompt | P1 verbatim; P2 = P1 + "keep each language in its own script, never translate"; P3 = P2 + glossary |
| Temperature | client default vs 0 |
| Audio prep | silence trimming with Silero VAD; loudness normalisation |
| Chunking | whole file vs 5 min chunks with overlap |
| Escalation | disputed spans re-sent to 3.8 Flash (20/day) |
| Combination | NIST SCTK `rover` across two finalists |

Prerequisite for prompts: the client must accept a full prompt and no forced `language=ru`
(today it appends to a fixed base prompt).

## Decision
Winner = lowest pooled normalised WER among setups whose daily limit covers my daily audio
hours; ties broken by name accuracy and reliability on a long diary.

## Prompt P1 (frozen text)
Stored in `benchmark/samples/120s-diary-ru/runs/_details/prompts-2026-10-14.json` ("p1-verbatim").

Note: the stored P1 says "in Russian Cyrillic" and the client adds "The audio is in ru language".
With Spanish and English clips, Phase 1 must use a language-neutral P1 ("keep each language in its
own script, never translate") and no forced language. This makes the full-prompt client change a
Phase 1 prerequisite.

## Answered
- Languages: Russian, English, Spanish.
- Volume: 1–3 h of diary audio per day — every Phase 1 bulk candidate fits.
