# Multimodal Transcription Core

Reusable Python transcription through Gemini, Groq Whisper, and configured
OpenAI-compatible endpoints. One library and CLI handle audio conversion,
chunking, reconstruction, explicit model selection/fallback, and shared limits.

## Quickstart

Python 3.10+ is required. Install into the environment that will use the core:

```sh
pip install "multimodal-transcription-core[live] @ git+https://github.com/elpinguinofrio/multimodal-transcription-core.git"
```

From a checkout, copy `transcription.example.ini` to `transcription.ini` **if you
do not already have a configuration**. Set the models, endpoints, paths, and
`[secrets] env_file` to an existing key file. Personal configuration and keys stay
out of Git. The CLI defaults to `./transcription.ini`; `--config` selects another.
`ffmpeg` and `ffprobe` are needed for conversion, splitting, and enhancement.

```sh
transcription-core recording.m4a --config transcription.ini
transcription-core recording.m4a --config transcription.ini -d notes/
transcription-core recording.m4a --config transcription.ini -m whisper-large-v3 --timestamps phrase --no-enhance
transcription-core recording.m4a --config transcription.ini -d notes/ --fuse gemini-3.5-flash-lite,whisper-large-v3
transcription-core --list-models --config transcription.ini
```

Use model names present in your configuration. `notes/` must already exist.
Without `-d`, text goes to stdout and diagnostics to stderr. With `-d`, files are
named `<name>.<model>.md`; reruns add `.(2)`, `.(3)`, etc., preserving earlier files.
`-m` selects exactly one model without fallback. Otherwise `[models] ranking`
controls selection and permitted fallback. `--list-models` makes provider metadata
requests; it does not transcribe audio. `--help` needs no configuration or provider.

The checkout's `./transcription-core` is an optional host/VM launcher backed by a
private runtime, not included in a fresh clone. See [host launcher](docs/host-launcher.md).
Ordinary installs use the package's `transcription-core` command.

## Library

```python
from multimodal_transcription_core import ModelSpec, RateLimiter, transcribe_file

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
transcription-core bench --config transcription.ini --audio benchmark/samples/120s-diary-ru/120s-diary-ru.wav --runs-dir benchmark/samples/120s-diary-ru/runs -m gemini-3.5-flash-lite -m whisper-large-v3 --label comparison
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
| `src/multimodal_transcription_core/` | Installable library and CLI; folder name is the Python import namespace |
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
`benchmark/samples/120s-diary-ru/120s-diary-ru.wav`. That recording is supplied for
inspection and reproduction of this benchmark. No general-purpose data license
is granted. Contact the owner for other reuse, redistribution, training or voice
synthesis. Public availability is not consent to impersonation.
