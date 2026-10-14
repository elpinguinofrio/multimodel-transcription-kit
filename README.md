# Multimodal Transcription Core

Reusable Python transcription for short and long audio through Gemini, Groq
Whisper, and OpenAI-compatible endpoints, with conversion, chunking, and merging.

## Russian sample benchmark

| Model | Verdict |
|---|---|
| **Gemini 3.8 Flash** | **Clear winner; nearly complete with minimal correction.** |
| GigaAM v3 E2E RNNT | Best readable local draft; substantial omissions. |
| Parakeet TDT 0.6B v3 | Useful timestamps; heavy text correction. |
| Whisper large-v3-turbo | Missing section and repetition failure. |
| Whisper large-v3 | Least usable result. |

Owner-reviewed qualitative comparison on one 120-second Russian-language sample
(`benchmark/samples/120s-diary-ru/`); it is not a general model ranking.

**Transcription quality cannot be scored by a script.** Speech is genuinely
ambiguous - overlapping words, filler, false starts and code-switching have no
single correct rendering, so two accurate transcripts can differ and both be
right. Character counts measure verbosity, not accuracy: a model that
hallucinates fluent text scores higher on length than one that correctly emits
nothing over silence. `scripts/score_transcripts.py` computes edit distance
against a human reference; it never tells you which transcript is better. That
needs a human listener or an LLM judge reading the audio alongside each
candidate.

Repeat runs land in a sample's gitignored `runs/` folder numbered `(1)`, `(2)`,
… so consistency is visible: parakeet returns byte-identical text run to run,
Gemini varied ~8% in length across two runs of the same audio.

## Use

```sh
pip install "git+https://github.com/elpinguinofrio/multimodal-transcription-core.git"
```

Command line (configuration: copy `transcription.example.ini`):

```sh
transcription-core recording.m4a --config transcription.ini          # transcript to stdout
transcription-core recording.m4a -d notes/ --config transcription.ini # notes/recording.<model>.md
transcription-core recording.m4a -m whisper-large-v3 --timestamps phrase --no-enhance ...
transcription-core recording.m4a -d notes/ --fuse gemini-3.5-flash-lite,whisper-large-v3 ...
transcription-core --list-models --config transcription.ini           # reachability, quota, key/model check
```

Library (the same path the command line runs):

```python
from multimodal_transcription_core import ModelSpec, RateLimiter, transcribe_file

limiter = RateLimiter.from_config("/path/to/transcription.ini")   # once per process
spec = ModelSpec(name="flash-lite", kind="gemini", model="gemini-3.5-flash-lite")
result = transcribe_file("recording.m4a", spec, language="ru", limiter=limiter)
print(result["text"], result["features"], result["warnings"])
```

Defaults on both: speaker labels and `*sound*` descriptions on, `major` timestamps
from provider alignment, automatic loudness/noise enhancement into a separate file,
long recordings split with overlap and rebuilt without duplicated boundary speech.
A feature a model cannot supply is dropped with a warning; one you ask for
explicitly raises `UnsupportedFeatureError`. Failures raise typed errors
(`TranscriptionError` subclasses); silence is reported as `no_speech`.
`transcribe_large()` and `transcribe()` remain for existing callers.

See [docs/integration.md](docs/integration.md) for shared rate limits and
migration notes, and [docs/operations.md](docs/operations.md) for dependencies,
retries and recovery.

Python 3.10+; conversion and splitting require `ffmpeg` and `ffprobe`. Code is
[MIT licensed](LICENSE); benchmark audio has [separate terms](#audio-terms).

## Audio terms

The MIT license covers code and documentation, **not**
`benchmark/samples/120s-diary-ru/120s-diary-ru.wav`. That recording is supplied
for inspection and reproduction of this benchmark. No general-purpose data
license is granted. Contact the owner for other reuse, redistribution, training
or voice synthesis. Public availability is not consent to impersonation.
