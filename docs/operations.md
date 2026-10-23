# Running unattended: dependencies, retries and recovery

What the core needs from the outside world, what it retries on its own, and what
needs a person. Check this list when something stops working after months untouched.

## External dependencies

| Dependency | Used for | If it is missing or changes |
|---|---|---|
| Python 3.10+, `requests`, `python-dotenv` | everything | install fails; pinned only by lower bounds in `pyproject.toml` |
| `ffmpeg` / `ffprobe` | duration, conversion, splitting, loudness assessment, enhancement, Live PCM | duration unknown: one request without chunking (warning); assessment skipped (warning); conversion needed: `AudioPreparationError` |
| `websockets` (`[live]` extra) | Gemini Live models only | `GeminiAPIError` naming the extra |
| Gemini API (`GEMINI_API_KEY`, optional `GEMINI_BASE_URL`) | Gemini REST and Live, chunk merges, fusion | see recovery below |
| Groq API (`GROQ_API_KEY`) | Whisper models | see recovery below |
| Your OpenAI-compatible server | `kind = http` models | unreachable: skipped by automatic selection; model missing from its `/v1/models` list: `--list-models` reports it unavailable |
| Local disk for `[limits] state_file` and `[usage] file` | shared queue and request history | SQLite on SMB/NFS is unreliable; keep both local |

## Bounded retries (no human needed)

- **429 / 5xx / connection errors** (Gemini, Groq, HTTP): up to `max_retries` (4) with exponential back-off
  capped at 60 s, honouring `Retry-After` and Gemini `RetryInfo`. With a shared limiter, a 429 makes every
  process wait, at most 20 rounds. A daily-quota 429 raises `QuotaExhausted` at once and blocks that model
  until Pacific midnight for every process sharing the state file.
- **Gemini Live**: a session the server ends (goAway, dropped socket) is retried once per part in a fresh
  session. Long recordings are split into parts of at most 10 minutes, one session each.
- **Nothing else is retried.** 400/401/403/404 are permanent and are not re-sent.

## Recovery without a person

- **Fallback:** without `-m`, the CLI (and `transcribe_best`) tries the next model in `[models] ranking` when
  a model is unreachable, out of quota, unable to supply an explicitly requested feature, or fails. Every
  skip and failure is written into the transcript header (`skipped:` / `fallback:`). With `-m` there is no
  fallback: the run exits 1 and says why.
- **Partial results are never lost:** `IncompleteTranscriptError.result` and `ReconstructionError.result`
  carry what was transcribed; `multimodel-transcription-kit bench` writes a receipt for every attempt, failed or not.

## What needs a person, and how you will notice

| Event | Symptom | Fix |
|---|---|---|
| A model is retired or renamed | `--list-models` shows `not served (retired, renamed or misspelled?)`; runs show `HTTP 404 ... model not served` and fall back | update `model =` in the config; update the tier file if limits changed |
| A key expires or is revoked | `--list-models` shows `... rejected`; runs show `credentials rejected` and fall back | replace the key in the env file named by `[secrets] env_file` |
| Quotas change | `QuotaExhausted` earlier or later than expected | edit `src/multimodel_transcription_kit/tiers/*.ini` or the model section |
| ffmpeg removed by an OS upgrade | `AudioPreparationError: ffmpeg is required` | reinstall ffmpeg |

Run `multimodel-transcription-kit --list-models --config <config>` after any of these; it checks each model with a
free metadata request (key accepted, model served) without transcribing anything. For `kind = http` it reads
the server's `/v1/models` list: a listed model proves the server knows the name, not which model answered a
given request. A server without that list is reported as answering but unverified.

## Release gate

Before publishing, stage the intended files, then check the exact Git-index
content that would be committed. There is no second public-file inventory to
maintain. Untracked files and unstaged edits are outside this release check.
Never force-add personal configuration, recordings, references, or transcripts.

```sh
python scripts/verify.py        # staged content: approved audio only; no private files or keys
python -m pytest -q             # offline: providers are mocked or a local synthetic server
python -m build --wheel        # build the installable package
```

Install the resulting wheel into a disposable consumer environment, then run
`multimodel-transcription-kit --help`, `multimodel-transcription-kit bench --help`, and
`multimodel-transcription-kit http --help` from outside the source tree. Check dependencies
with `python -m pip check`. Keep shared host/VM validation artifacts under the
shared root, in a new ignored `.private/` run directory.

The verifier reads staged text, audio, and `benchmark/manifest.json` directly from
Git. A cleaned working copy cannot hide a staged credential or altered recording.
The benchmark manifest remains necessary: it identifies the approved recording,
its hash, exact duration, and terms.

Offline checks do not contact real providers. Real-provider and consuming-app
checks are run separately and reported as such.
