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

Owner-reviewed qualitative comparison on one 120-second Russian-language sample;
it is not a general model ranking.

## Use

```sh
pip install "git+https://github.com/elpinguinofrio/multimodal-transcription-core.git"
```

```python
from multimodal_transcription_core import transcribe_large

result = transcribe_large("recording.m4a")
```

Python 3.10+; conversion and splitting require `ffmpeg` and `ffprobe`. Code is
[MIT licensed](LICENSE); benchmark audio has [separate terms](benchmark/DATA_LICENSE.md).
