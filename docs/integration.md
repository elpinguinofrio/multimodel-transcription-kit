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

[model.gemini-3.5-flash-lite]
kind = gemini
rpm = 15                                     # values copied from AI Studio
tpm = 250K                                   # "250K", "14.4K", "Unlimited" all parse
rpd = 500
```

The limiter only queues models that have `rpm`, `tpm` or `rpd` set.
A model with no limits configured still gets 429 retries, but no proactive queuing.
A limit of `0` (shown in AI Studio for models your tier doesn't include) stops requests with `QuotaExhausted`.

## 2. Install into each app's environment (host or VM)

```sh
# CORE = this checkout. In the VM: /Volumes/VM-shared/projects/multimodal-transcription-core
# On the host, use the Mac's own path to the same folder.
CORE=/Volumes/VM-shared/projects/multimodal-transcription-core
<app>/.venv/bin/pip install "$CORE"          # or: pip install dist/multimodal_transcription_core-0.2.0-py3-none-any.whl
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

### Which model each function uses

Always pass `model=` with a model that has limits in your config.

- `transcribe(..., model=...)` uses the model you pass. Without one it uses `gemini-3-flash-preview` (20 per day on this key).
- `transcribe_large(..., model=...)` uses that one model for everything: files under 30 minutes, each chunk of a longer file, and the merges between chunks.
  Without `model` it keeps the old behaviour: `transcribe()`'s default for short files, and `gemini-3.5-flash` for chunks and merges. That model has no limits in the config, so it's only covered by 429 handling.

## 4. Host and VM together

SQLite state can only coordinate processes on **one machine**.
The VM and the host each have their own state file, so each can spend whatever limits its own config holds.
If both use the same API key:

- **Simplest:** transcribe on one machine only.
- **Or split every limit** between the two configs so the totals stay at the AI Studio values. For example, host `rpm 10 / tpm 170K / rpd 350` and VM `rpm 5 / tpm 80K / rpd 150`.

If they overlap anyway, the server's 429 is handled: it waits as long as the server asks, and a daily-quota 429 stops with `QuotaExhausted`.

## 5. Per app

- **Exocortex:** keeps its own `engine/lib/transcriber` until its migration agent validates the installed core through the real app entry point (AGENTS.md).
  The swap: replace the `_load_module(.../transcriber.py)` and `chunked_transcriber.py` loads in `engine/pipelines/voice/pipeline.py` with the imports above.
  Then pass `api=GeminiAPI(rate_limiter=limiter)`, catch `QuotaExhausted` to write a "retry after" sidecar, and only then delete the copy.
- **Quickstart example** (`examples/quickstart/run.py`): builds one limiter with `RateLimiter.from_parser(cfg, section_prefix="provider.")`.
  It reads `rpm`/`tpm`/`rpd` from each `[provider.*]` section and `[limits]` from its `config.ini`. Any other caller is protected only if it passes a `RateLimiter`.
- **Clip Smith:** doesn't call the core today. Its transcription is local whisper plus `scripts/transcribe/correct_transcript_gemini.py`.
  Any new cloud transcription step should use the snippet above with the shared config, not a new Gemini client.

## Migration notes: 0.1.0 → 0.2.0

- New, optional: `rate_limiter=` on `GeminiAPI` and `GroqAPI`; `api=` on `transcribe` and `transcribe_large`; `model=` on `transcribe_large`. New exports: `RateLimiter`, `Limits`, `RateLimitError`, `QuotaExhausted`, `RateLimitTimeout`, `RequestTooLarge`.
- **Changed:** an exhausted 429 now raises `RateLimitTimeout` instead of returning a 429, which callers saw as `GeminiAPIError("API error 429")`. A daily-quota 429 raises `QuotaExhausted` right away.
  `transcribe()` no longer wraps these in `RuntimeError`.
- **Changed:** `GeminiAPI.generate_text` and the chunk merge now use the same retry policy and limiter as transcription.
  A failed boundary merge keeps both sides of the boundary instead of dropping that text.
- **Groq:** retries 429s, honouring `retry-after`. It is proactively limited only if you set `rpm`/`rpd` for it, and its real limits are not recorded yet.
  The local day count resets at Pacific midnight, which may differ from Groq's reset time.
