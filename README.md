# Multimodel Transcription Kit

Reusable Python transcription through Gemini, Groq Whisper, and configured
OpenAI-compatible endpoints. One library and CLI handle audio conversion,
chunking, reconstruction, explicit model selection/fallback, and shared limits.

## Quickstart

Python 3.10+ is required. From this checkout, install into the environment that will use the kit:

```sh
python -m pip install ".[live]"
```

From a checkout, copy `transcription.example.toml` to `transcription.toml` **if you
do not already have a configuration**. It has three parts:

- **Providers**: one `[[aistudio]]`, `[[antigravity]]` or `[[openai]]` block per
  account, each with your own `name` and optional `key_env`, `base_url` and
  `costs_money = true`. A proxy is just an account with its own `base_url`. A local
  speech server is an `[[openai]]` account without a key. `parallel_chunks = N`
  sends up to N parts of a long recording at once (default 1, one after another).
- **Models**, best first: one `[[model]]` block each, listing its `providers` in the
  order to try. `word_times = true` marks the models that may supply word times.
  `chunk_seconds` is the longest part one request carries; longer recordings are split.
  The example sets it for every model; `gemini-3.5-flash-lite` gets 600 (10 minutes),
  since its 30-minute parts came back incomplete.
- **Profiles** (optional): named model orders, one `[[profile]]` block each with
  `name` and `models`, best first. See [Quality or cost](#quality-or-cost).

### Quality or cost

One config can hold several model orders, so each caller picks what its work needs:

| Profile | For | First model | Trade-off |
|---|---|---|---|
| `quality` | content: clips, videos, anything published | `gemini-3.8-flash` | preferred in the [owner's review](#russian-sample-benchmark); slower and pricier |
| `balanced` | bulk: hundreds of hours of meetings, notes | `gemini-3.5-flash-lite` | cheaper and faster; less accurate in the owner's tests |

```sh
multimodel-transcription-kit meeting.m4a --profile balanced -d notes/
```

`[defaults] profile` names the one used when a caller passes no `--profile`; the example
sets `quality`, so callers get the most accurate order unless they ask for less. Without
any `[[profile]]`, the `[[model]]` block order is the order. A profile uses only the models
it lists, and paid routes still come after every free one. Word-time models keep the
`[[model]]` block order.

Every free route of every model is tried before any `costs_money` account. Set
`[secrets] env_file` to an existing key file; personal configuration and keys stay out
of Git. The CLI uses `./transcription.toml` when present, else `./transcription.ini`,
and `--config` selects another file. `--convert-config transcription.ini` prints the
TOML equivalent of an older INI config. Reading TOML needs Python 3.11+ (or `tomli`).
`ffmpeg` and `ffprobe` are needed for conversion, splitting, and enhancement.

```sh
multimodel-transcription-kit recording.m4a --config transcription.ini
multimodel-transcription-kit recording.m4a --config transcription.ini -d notes/
multimodel-transcription-kit recording.m4a --config transcription.ini -m whisper-large-v3 --timestamps phrase --no-enhance
multimodel-transcription-kit recording.m4a --config transcription.ini -d notes/ --fuse gemini-3.5-flash-lite,whisper-large-v3
multimodel-transcription-kit --list-models --config transcription.ini
```

Use model names present in your configuration. `notes/` must already exist.
Without `-d`, text goes to stdout and diagnostics to stderr. With `-d`, files are
named `<name>.<model>.md`; reruns add `.(2)`, `.(3)`, etc., preserving earlier files.
`-m` selects exactly one model (trying its providers in order). Otherwise the profile's
order (see above) controls selection and permitted fallback. When one part of a long
recording fails (an error, or `[SILENCE]` over audible speech), only that part goes to the next
models in the order; the other parts keep their text. `--list-models` makes provider metadata
requests; it does not transcribe audio. `--help` needs no configuration or provider.

The checkout's `./multimodel-transcription-kit` is an optional host/VM launcher backed by a
private runtime, not included in a fresh clone. See [host launcher](docs/host-launcher.md).
Ordinary installs use the package's `multimodel-transcription-kit` command.

## Library

```python
from multimodel_transcription_kit import ModelSpec, RateLimiter, transcribe_file

limiter = RateLimiter.from_config("/path/to/transcription.ini")
spec = ModelSpec(name="flash-lite", kind="gemini", model="gemini-3.5-flash-lite")
result = transcribe_file("recording.m4a", spec, language="ru", limiter=limiter)
print(result["text"], result["features"], result["warnings"])
```

The caller supplies configuration and credentials; importing the library does
not discover private settings, call providers, or write application data.
`transcribe_large()` and `transcribe()` remain for existing callers.

Speaker labels, sound descriptions, and `major` timestamps are requested by
default; timestamps require provider alignment. An unsupported default produces
a warning; an explicitly requested unsupported feature raises
`UnsupportedFeatureError`. Automatic enhancement uses a temporary copy, removed
after the run unless `--keep-enhanced` is supplied. The original is preserved.
Failures raise typed errors; successful silence is reported as `no_speech`.

## Compare models

Use the same CLI and root configuration for benchmarking, rather than a separate
example application:

```sh
multimodel-transcription-kit bench --config transcription.ini --audio benchmark/samples/120s-diary-ru/120s-diary-ru.wav --runs-dir benchmark/samples/120s-diary-ru/runs -m gemini-3.5-flash-lite -m whisper-large-v3 --label comparison
```

Each attempt records a receipt in `_details/receipts/`, including failures.
Successful transcripts are `<model>__<label>__v001.txt`, then `v002`, etc.
Runs stay local and gitignored. Model names above must be configured
for the intended route; benchmarking contacts those providers.

### Russian sample benchmark

Historical owner-reviewed qualitative comparison on one 120-second sample;
this is not a general model ranking or a current availability check.

| Model | Owner's assessment |
|---|---|
| Gemini 3.8 Flash | Nearly complete, minimal correction; preferred on this sample |
| GigaAM v3 E2E RNNT | Best readable local draft; substantial omissions |
| Parakeet TDT 0.6B v3 | Useful timestamps; heavy text correction |
| Whisper large-v3-turbo | Missing section and repetition failure |
| Whisper large-v3 | Least usable result |

`scripts/score_transcripts.py` computes edit distance, not a complete judgment of
transcription quality. Accuracy claims require a supplied and reviewed human
reference; private references and transcripts are never published.

## Project layout

| Path | Purpose |
|---|---|
| `src/multimodel_transcription_kit/` | Installable library and CLI; folder name is the Python import namespace |
| `tests/` | Offline tests with mocked providers or synthetic local servers |
| `docs/` | Integration, operations, usage history, and roadmap |
| `benchmark/` | Approved sample, hash manifest, compatibility entry point |
| `scripts/` | Release checks, transcript scoring, portable-runtime tooling |
| `skills/transcribe/` | Agent workflow using the existing CLI |
| `.private/` | Ignored local runtime, state, builds, and validation artifacts |

[Product specification](docs/PRODUCT.md) is the owner-controlled specification.
See [integration and migration](docs/integration.md),
[operations and release checks](docs/operations.md),
[usage history](docs/usage-history.md), [Live transcription](docs/live-transcription.md),
and the [roadmap](docs/roadmap.md). Older implementation plans under
`docs/superpowers/` are historical context, not current setup instructions.

## Audio terms

Code and documentation are [MIT licensed](LICENSE). The license does **not** cover
`benchmark/samples/120s-diary-ru/120s-diary-ru.wav` or
`benchmark/samples/101s-codexbar-ru/101s-codexbar-ru.wav`. These recordings are supplied for
inspection and reproduction of this benchmark. No general-purpose data license
is granted. Contact the owner for other reuse, redistribution, training or voice
synthesis. Public availability is not consent to impersonation.
