# Simple request usage history

The core can keep a local SQLite history for Gemini transcription and text
generation (including chunk merges), Gemini Live sessions and model discovery,
Groq transcription, and every OpenAI-compatible HTTP transcription endpoint.
The log is independent of the limiter: it also works without quota limits.

Enable it in the CLI or quickstart configuration:

```ini
[usage]
file = ~/.local/state/multimodel-transcription-kit/usage.sqlite
```

Use a local disk, not SMB/NFS. Give every application on one machine the same
file to combine its history. The VM and host have separate files. Existing
private configuration files are not automatically modified; no logging occurs
when the section is absent. The library discovers no configuration itself.

Fetch totals, or individual requests, as JSON:

```sh
multimodel-transcription-kit usage --config /path/to/transcription.ini
multimodel-transcription-kit usage --config /path/to/transcription.ini --history
multimodel-transcription-kit usage --db /path/to/usage.sqlite --provider gemini --since 2026-10-14T00:00:00Z
```

History is oldest first, with `--limit 100` and `--offset 0` defaults. Increase
the offset to fetch later pages (maximum page size 10,000). Summaries cover all
matching rows, not just the history page. `--model`, `--provider`, `--since`
(inclusive) and `--until` (exclusive) filter either view. Date filters require
a timezone. Querying a missing database reports an error and creates nothing.
Queries do not load secrets or contact providers.

For a future UI, use the same Python API:

```python
from multimodel_transcription_kit import UsageLog, GeminiAPI, GroqAPI

usage = UsageLog("/path/to/usage.sqlite")
gemini = GeminiAPI(usage_log=usage)  # also accepts rate_limiter= independently
groq = GroqAPI(usage_log=usage)
# transcribe_http(audio, endpoint, output, usage_log=usage)

totals = usage.summary(provider="gemini")
requests = usage.requests(limit=100, offset=0)
```

For the raw HTTP CLI, pass `--usage-file /path/to/usage.sqlite` to
`multimodel-transcription-kit http`. Every direct library caller must supply `usage_log=`;
passing only a limiter does not enable usage recording. Gemini Live uses the
log supplied to its `GeminiAPI`.

Each completed HTTP attempt is one row, including 429/5xx and connection
failures. Retries share a `request_id`, with attempt numbers starting at 1.
A Live session counts as one request, not one per audio packet. Quota stops
before a network send do not count as provider requests. A process terminated
before the attempt finishes may leave no history row. There is no retroactive
import from Google's dashboard or from past requests outside this core.

Rows contain UTC start time, provider, requested model, operation, duration,
HTTP status if applicable, error type, finish reason, reported model version,
and reported input/output/total/thought/cached token counts. No keys, URLs,
audio paths, audio, prompts, transcripts or server error messages are stored.
`status` describes the transport attempt (`ok`, `error`, or a recognized
Gemini `blocked` finish reason); HTTP 200 alone does not establish transcript
quality. Gemini model discovery uses model `*` and operation `models`.

Token usage is actual provider-reported metadata, never the limiter's audio
token estimate. Unreported or malformed counts remain `null`, not zero.
Groq Whisper and local servers often do not report tokens. Summaries add known
values and expose `unknown_input_requests`, `unknown_output_requests`, etc.,
so incomplete totals remain visible. Total tokens may include thinking and
other provider-specific accounting; do not assume input plus output equals
the reported total. A reported zero remains zero.

`requests` counts attempts; `calls` counts distinct logical requests;
`retries` counts extra attempts. `succeeded` and `failed` count attempt statuses,
so a recovered 429 contributes one failed attempt and one successful attempt.
History has no automatic pruning. SQLite write failures surface to the caller;
they are not silently discarded. An error after sending can mean the provider
processed the request even if the history could not be saved.

Migration: `usage_log=` is optional and appended to the existing client and
HTTP signatures. Existing callers retain their return formats. Default long
file chunks, merges and `GeminiAPI.generate_text()` now use
`gemini-3.6-flash`; explicit model choices and short-file defaults are unchanged.
Exocortex and Clip Smith must pass a `UsageLog` when adopting the updated core.
