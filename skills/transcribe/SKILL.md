---
name: transcribe
description: Transcribe an audio or video recording to text with the installed multimodel-transcription-kit, using the best available configured model. Use when the user asks to transcribe a recording, get its text, or save a transcript.
---

# Transcribe a recording

Runs the `multimodel-transcription-kit` command from the installed `multimodel-transcription-kit`
package. Model choice, fallback, chunking, conversion, enhancement and rate limits are
the library's job; do not reimplement them.

## Before running

1. Establish the current machine with `uname -n` and resolve the user's shared checkout
   from this machine. A missing PATH entry does not establish that the tool is absent.
   Try the installed command, then an explicit checkout-local launcher. Never activate a
   VM venv on the host: its Python paths can belong to the VM even on a shared disk.
2. The checkout's `./multimodel-transcription-kit` launcher uses its private portable runtime,
   requiring no host Python, pip, uv, PATH registration or VM command connection.
   Run it with `--portable-info` and `--help` to confirm the runtime and paths. For video,
   also check the reported `ffmpeg` and `ffprobe`; startup alone does not prove transcription.
   On the VM an already working native `.venv/bin/multimodel-transcription-kit` can still be used.
3. Respect an explicit config. The portable launcher otherwise supplies its packaged
   configuration with paths resolved under the current shared root; do not pass a
   VM-only `~/.config/transcription` path on the host. A native installation uses the
   caller's config or the machine's existing config. Never print credentials or silently
   install dependencies, edit config, start SSH, or switch transcription providers.
4. Use the resolved executable for the optional metadata check `--list-models`.
   With a native installation pass the explicit config; the portable launcher can use
   its packaged default. Report host execution as untested until exercised on the host.

## Run

- Text to the conversation: `multimodel-transcription-kit "$AUDIO" --config "$CONFIG"`
- Saved transcript: `multimodel-transcription-kit "$AUDIO" --config "$CONFIG" -d "$DIR"`
  writes `<name>.<model>.md` beside earlier runs (`(2)`, `(3)`, ...); nothing is overwritten.

Here `multimodel-transcription-kit` means the executable resolved above, which may be an
absolute checkout path. Omit `--config` when using the portable packaged default.

Pass only what the user asked for: `-m <model>` (exactly that model, no fallback),
`--language xx`, `--no-speakers`, `--no-sounds`, `--timestamps none|major|phrase`,
`--no-enhance`, `--keep-enhanced`, `--prompt TEXT`, `--temperature T`, `--overlap-seconds S`,
`--max-retries N`, `--timeout S`, `--receipt` (with `-d`), `--fuse a,b` (with `-d`, runs several models).

## Report

- Exit 0: give the transcript or the written path (printed on stderr). Mention any
  `warning:` lines (e.g. a feature the model cannot supply) and `no speech detected`.
- Exit 1: transcription failed; quote the stderr lines naming each model tried and why.
- Exit 2: usage or configuration problem; quote the message.
