# Quickstart: compare providers on one recording

A small application that transcribes one audio file with every provider you
enable, then prints a comparison table. It exists to show the intended split:

> **the application owns configuration and secrets; the library reads neither.**

`multimodal_transcription_core` never looks for a config file, never walks up
directories hunting for `.env`, and never picks a provider for you. This example
does all three, and hands the library explicit values.

## Run it

```sh
pip install -e ".[dev]"              # from the repo root
cd examples/quickstart
cp .env.example ~/.config/transcription/.env   # add your keys there
$EDITOR config.ini                   # set [audio] path, enable providers
python run.py
```

Output:

```
provider   model                         chars    secs  status
local      parakeet-tdt-0.6b-v3          1,001     2.1  ok
groq       whisper-large-v3                690     2.2  ok
gemini     gemini-3-flash-preview          886    24.8  ok
groq-turbo whisper-large-v3-turbo          780     2.2  ok
```

A provider that fails is reported in the table and the others still run.

## Nothing is ever lost

Every invocation writes to its own directory under `[output] dir` (default
`./transcripts`, gitignored), named by UTC timestamp. Existing runs are never
touched, and **failed attempts are recorded with their error text** - a
transient 503 halfway through a long run stays on disk for analysis instead of
vanishing with the terminal.

```
examples/quickstart/transcripts/         # gitignored, never committed
├── runs.jsonl                           # append-only, one line per attempt, all runs
└── 20260914T012647Z/
    ├── run.json                         # manifest: audio sha256, per-provider result
    ├── sample-120s-diary__parakeet-tdt-0.6b-v3.txt
    └── ...
```

Each record carries `run_id`, `provider`, `model`, `endpoint`, `language`,
`audio`, `audio_sha256`, `started_utc`, `seconds`, and either
`chars`/`words`/`transcript` or `error_type`/`error`. The audio hash makes runs
comparable across time even if a file is moved or renamed.

## config.ini

| Section | Purpose |
|---|---|
| `[secrets]` | `env_file` — path to a `.env` with your keys. Never store keys in this repo. |
| `[audio]` | `path` to the recording, `language` hint. The file is never copied. |
| `[output]` | Parent `dir` for run directories, default `./transcripts`. Gitignored. |
| `[provider.<name>]` | `enabled`, `kind` (`http`/`groq`/`gemini`), `model`, and `endpoint` for `http`. |

Add a provider by adding a section — no code change for `http`, since any
OpenAI-compatible transcription endpoint works:

```ini
[provider.my-server]
enabled  = true
kind     = http
endpoint = http://my-box:8765/v1/audio/transcriptions
model    = parakeet-tdt-0.6b-v3
```

## What it demonstrates

- Secrets are loaded by the app via `TRANSCRIPTION_ENV_FILE`, the library's only
  explicit opt-in — see `load_secrets()`.
- Provider choice is data (`config.ini`), not code.
- One failing provider never aborts the run.
- Nothing is written outside `[output] dir`; audio is read, never copied.
