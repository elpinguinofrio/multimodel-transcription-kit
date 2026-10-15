# Using the core from Clip Smith, Exocortex or any other app

The core is what queues requests and enforces rate limits, so your apps don't need any limit logic of their own.
Each app makes one `RateLimiter` per process from a shared config file and passes it to the client.
Every process on the same machine that names the same state file shares one queue.

## 1. One config per machine

Keep one file at `~/.config/transcription/transcription.ini` on each machine that transcribes.
Start from the repo's `transcription.example.ini`; a `transcription.ini` in the repo root is gitignored.
The library never searches for this file. Your app passes its path explicitly.

```ini
[secrets]
env_file = ~/.config/transcription/.env      # GEMINI_API_KEY=..., GROQ_API_KEY=...

[limits]
# Local disk only. Never /Volumes/VM-shared: SQLite locking is unreliable on SMB.
state_file = ~/.local/state/transcription-core/ratelimit.sqlite
max_wait_seconds = 3600                      # longest a request may wait for a slot

tiers = gemini-pro-tier                      # public tier limits shipped with the package

[model.gemini-3.5-flash-lite]
kind = gemini
# rpd = 250                                  # optional: this machine's share, overrides the tier
```

The limits preset lives in the package (`src/multimodal_transcription_core/tiers/gemini-pro-tier.ini`). Its source is the owner's AI Studio project snapshot, not a public quota guarantee for every Google AI Pro subscriber. Check your project's active quotas before using it.
Your config only names the tier and holds what is specific to one machine.
`rpm`/`tpm`/`rpd` in a `[model.*]` section override the tier one field at a time. Values like `"250K"`, `"14.4K"` and `"Unlimited"` all parse.
If Google changes the quotas, update the tier file and commit it.

The limiter only queues models that have limits from a tier or the config.
A model with no limits configured still gets 429 retries, but no proactive queuing.
A limit of `0` (shown in AI Studio for models your tier doesn't include) stops requests with `QuotaExhausted`.

## 2. Install into each app's environment (host or VM)

```sh
# CORE = this checkout. In the VM: /Volumes/VM-shared/projects/multimodal-transcription-core
# On the host, use the Mac's own path to the same folder.
CORE=/Volumes/VM-shared/projects/multimodal-transcription-core
<app>/.venv/bin/pip install "$CORE"          # or: pip install dist/multimodal_transcription_core-0.3.0-py3-none-any.whl
<app>/.venv/bin/transcription-core --list-models --config ~/.config/transcription/transcription.ini
```

`--list-models` should list your models with `today N/500`. The count covers only requests made through the core on this machine.
AI Studio's rate-limit page stays the source of truth.

## 3. Call it

```python
import os
from pathlib import Path
from multimodal_transcription_core import (
    GeminiAPI, RateLimiter, QuotaExhausted, RateLimitTimeout, RequestTooLarge,
    transcribe, transcribe_large,
)

CONFIG = Path("~/.config/transcription/transcription.ini").expanduser()
# The app names its key file explicitly; the library never searches for one.
os.environ.setdefault("TRANSCRIPTION_ENV_FILE", str(Path("~/.config/transcription/.env").expanduser()))
limiter = RateLimiter.from_config(CONFIG)   # once per process, reused for every call
api = GeminiAPI(rate_limiter=limiter)

try:
    result = transcribe("clip.m4a", model="gemini-3.5-flash-lite", api=api)   # one request
    # result = transcribe_large("long.m4a", api=api, model="gemini-3.5-flash-lite")  # any length
except QuotaExhausted as err:      # the day's quota is used up; err.resets_at = Pacific midnight
    ...                            # mark the job "retry after err.resets_at"; don't retry now
except RateLimitTimeout:           # no slot within max_wait_seconds, or 429s never stopped
    ...                            # re-queue the job later
except RequestTooLarge:            # one clip needs more tokens than the model allows per minute
    ...                            # split the audio or use a model with a bigger TPM
```

The three errors above all inherit from `RateLimitError`.
They are never wrapped in `RuntimeError`, and `transcribe_large` raises them instead of merging around a missing chunk.
For Groq, pass the same limiter: `GroqAPI(rate_limiter=limiter)`.

To keep request history and actual input/output token usage across all providers,
set `[usage] file` in the CLI configuration or pass a shared `UsageLog` to each
client. Fetch totals with `transcription-core usage --config <config>`.
See [Simple request usage history](usage.md) for setup, filters and the Python API.

### Which model each function uses

Always pass `model=` with a model that has limits in your config.

- `transcribe(..., model=...)` uses the model you pass. Without one it uses `gemini-3-flash-preview` (20 per day on this key).
- `transcribe_large(..., model=...)` uses that one model for everything: files under 30 minutes, each chunk of a longer file, and the merges between chunks.
  Without `model` it uses `transcribe()`'s default for short files, and `gemini-3.6-flash` for chunks and merges. `GeminiAPI.generate_text()` also defaults to 3.6 Flash. Its preset contains 5 RPM, 250K TPM and 20 RPD; load the preset and pass the limiter to enforce them. The old 3.5 Flash enum remains available for compatibility, but Google routes it to 3.6 Flash.

## 4. Host and VM together

SQLite state can only coordinate processes on **one machine**.
The VM and the host each have their own state file, so each can spend whatever limits its own config holds.
If both use the same API key:

- **Simplest:** transcribe on one machine only.
- **Or split every limit** between the two configs so the totals stay at the tier values. For example, host `rpm 10 / tpm 170K / rpd 350` and VM `rpm 5 / tpm 80K / rpd 150`, set as overrides in each machine's `[model.*]` section.

If they overlap anyway, the server's 429 is handled: it waits as long as the server asks, and a daily-quota 429 stops with `QuotaExhausted`.

## 5. Per app

- **Exocortex:** keeps its own `engine/lib/transcriber` until its migration agent validates the installed core through the real app entry point (AGENTS.md).
  The swap: replace the `_load_module(.../transcriber.py)` and `chunked_transcriber.py` loads in `engine/pipelines/voice/pipeline.py` with the imports above.
  Then pass `api=GeminiAPI(rate_limiter=limiter)`, catch `QuotaExhausted` to write a "retry after" sidecar, and only then delete the copy.
- **Quickstart example** (`examples/quickstart/run.py`): builds one limiter with `RateLimiter.from_parser(cfg, section_prefix="provider.")`.
  It reads `[limits] tiers` and any `rpm`/`tpm`/`rpd` overrides in each `[provider.*]` section of its `config.ini`. Any other caller is protected only if it passes a `RateLimiter`.
- **Clip Smith:** doesn't call the core today. Its transcription is local whisper plus `scripts/transcribe/correct_transcript_gemini.py`.
  Any new cloud transcription step should use the snippet above with the shared config, not a new Gemini client.

## Migration notes: 0.2.0 → 0.3.0

Both apps must review these before upgrading; signatures are unchanged, behaviour is not.

- **New primary API, same path as the CLI:** `transcribe_file(audio, ModelSpec(...), ...)` for any provider kind
  (Gemini REST and Live, Groq, OpenAI-compatible HTTP), `transcribe_best(...)` (ranking + fallback),
  `transcribe_fused(...)`, `verify_model(spec)`. `ModelSpec.from_options(name, dict(config["model.<name>"]))`
  builds a spec from a config section.
- **Changed: failures raise.** `GeminiAPI.speech_to_text` raises `ResponseError` (also a `GeminiAPIError`)
  for blocked, empty, malformed and truncated (`MAX_TOKENS`) responses and for whole-response repetition
  loops, instead of returning `[PARSE_ERROR] ...` / `[HALLUCINATION]` text with `success=True`.
  `GeminiAPI.generate_text` raises on a truncated generation instead of returning the partial text.
- **Changed: long files never return holes as success.** `transcribe_large` raises
  `IncompleteTranscriptError` when any part failed; `error.result` holds the partial transcript (with
  `[Part N - TRANSCRIPTION FAILED]` markers, `success=False`) and `error.failed_parts`. All parts failing
  still re-raises the first part's original error. A boundary that cannot be joined safely raises
  `ReconstructionError` instead of concatenating (which duplicated overlap). The `[Parts 1-3]` headers are gone.
- **Changed: unknown usage is `None`.** `input_tokens`/`output_tokens` are the provider's reported values or
  `None`; they are no longer zero-filled. Code that sums them must skip `None`.
- **New result keys:** `no_speech` (explicit silence), `provider`, `plain_text`, `segments`, `features`,
  `settings`, `audio` (source hash, loudness assessment, enhanced file), `warnings`, `boundaries`, `client`.
- **Errors:** every typed error derives from `TranscriptionError`, which derives from `RuntimeError`, so
  `except RuntimeError` still catches them. `GeminiAPIError` and `GroqAPIError` now derive from it too and
  carry `.status` (HTTP status) when there was one. Rate-limit errors are unchanged and never wrapped.
- **Changed: `transcribe_http` / `transcription-core http`** now use the shared retry policy (429/5xx) and
  accept `limiter=`; a non-200 raises `HTTPProviderError`, which is still a `urllib.error.HTTPError`.
- **Changed: chunk boundaries** are spread evenly, so no part exceeds `chunk_sec + overlap_sec` (the last
  part could reach 1.5x before). Starts stay on round numbers when the duration is a multiple of the chunk.
- **CLI:** new flags `--speakers/--no-speakers`, `--sounds/--no-sounds`, `--timestamps none|major|phrase`,
  `--enhance/--no-enhance`, `--fuse MODELS`, `--receipt`; `bench` subcommand. Without `-m` a failing model
  falls back down the ranking; with `-m` it exits 1. Saved documents add `model:`, `features:`, `settings:`,
  `audio:`, `fallback:` and `warning:` lines after `src:`. Exit codes: 0 ok, 1 transcription failed,
  2 usage or configuration error.
- **Enhancement is automatic** (owner decision): problem audio gets `<stem>.enhanced-<id>.flac` beside the
  source, and that copy is transcribed. Pass `enhance=False` (`--no-enhance`) to keep apps' current behaviour.
- `[defaults] prompt` in the CLI config applies to Gemini models only (Whisper prompts are context, not
  instructions); a model section's own `prompt` applies to that model.
- **Gemini prompts are the whole instruction.** A `ModelSpec.prompt` / `transcribe_file(prompt=...)` for a
  Gemini model replaces the built-in "Transcribe the following audio..." text instead of being appended to it,
  and "The audio is in <language> language." is no longer added (write the language into the prompt). Only
  the speaker and sound sentences are still added when those features are on. Without a prompt, the built-in
  text and the language sentence are unchanged. The low-level `GeminiAPI.speech_to_text(prompt=...)` still
  appends; its new `full_prompt=` argument is what replaces.

## Migration notes: 0.1.0 → 0.2.0

- New, optional: `rate_limiter=` on `GeminiAPI` and `GroqAPI`; `api=` on `transcribe` and `transcribe_large`; `model=` on `transcribe_large`. New exports: `RateLimiter`, `Limits`, `RateLimitError`, `QuotaExhausted`, `RateLimitTimeout`, `RequestTooLarge`.
- **Changed:** an exhausted 429 now raises `RateLimitTimeout` instead of returning a 429, which callers saw as `GeminiAPIError("API error 429")`. A daily-quota 429 raises `QuotaExhausted` right away.
  `transcribe()` no longer wraps these in `RuntimeError`.
- **Changed:** `GeminiAPI.generate_text` and the chunk merge now use the same retry policy and limiter as transcription.
  A failed boundary merge keeps both sides of the boundary instead of dropping that text.
- **Groq:** retries 429s, honouring `retry-after`. It is proactively limited only if you set `rpm`/`rpd` for it, and its real limits are not recorded yet.
  The local day count resets at Pacific midnight, which may differ from Groq's reset time.
