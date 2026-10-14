---
name: transcribe
description: Transcribe an audio or video recording to text with the installed multimodal-transcription-core, using the best available configured model. Use when the user asks to transcribe a recording, get its text, or save a transcript.
---

# Transcribe a recording

Runs the `transcription-core` command from the installed `multimodal-transcription-core`
package. Model choice, fallback, chunking, conversion, enhancement and rate limits are
the library's job; do not reimplement them.

## Before running

1. `command -v transcription-core` must succeed. If not, say the package is not installed
   in this environment; do not install it without the user's approval.
2. Find the config: the path the user gave, else `~/.config/transcription/transcription.ini`.
   If neither exists, ask for it. Never edit it and never print the env file it points to.
3. Optional health check (no transcription, no quota): 
   `transcription-core --list-models --config "$CONFIG"`

## Run

- Text to the conversation: `transcription-core "$AUDIO" --config "$CONFIG"`
- Saved transcript: `transcription-core "$AUDIO" --config "$CONFIG" -d "$DIR"`
  writes `<name>.<model>.md` beside earlier runs (`(2)`, `(3)`, ...); nothing is overwritten.

Pass only what the user asked for: `-m <model>` (exactly that model, no fallback),
`--language xx`, `--no-speakers`, `--no-sounds`, `--timestamps none|major|phrase`,
`--no-enhance`, `--receipt` (with `-d`), `--fuse a,b` (with `-d`, runs several models).

## Report

- Exit 0: give the transcript or the written path (printed on stderr). Mention any
  `warning:` lines (e.g. a feature the model cannot supply) and `no speech detected`.
- Exit 1: transcription failed; quote the stderr lines naming each model tried and why.
- Exit 2: usage or configuration problem; quote the message.
