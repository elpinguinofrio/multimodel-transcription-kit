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

```python
from multimodal_transcription_core import transcribe_large

result = transcribe_large("recording.m4a")
```

To stay under your API limits across apps, pass a shared `RateLimiter`; see
[docs/integration.md](docs/integration.md).

Python 3.10+; conversion and splitting require `ffmpeg` and `ffprobe`. Code is
[MIT licensed](LICENSE); benchmark audio has [separate terms](#audio-terms).

## Audio terms

The MIT license covers code and documentation, **not**
`benchmark/samples/120s-diary-ru/120s-diary-ru.wav`. That recording is supplied
for inspection and reproduction of this benchmark. No general-purpose data
license is granted. Contact the owner for other reuse, redistribution, training
or voice synthesis. Public availability is not consent to impersonation.
