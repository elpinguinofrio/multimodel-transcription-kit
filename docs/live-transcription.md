# Gemini Live file transcription

Install the core with its optional `live` extra into each consumer environment.
The transport dependency is pinned to `websockets==15.0.1`; the ordinary upload
client does not need that extra. Non-native audio requires an existing `ffmpeg`.

```python
from multimodel_transcription_kit import GeminiAPI, transcribe_live

api = GeminiAPI(rate_limiter=limiter)  # reuse the application's configured client
result = transcribe_live("recording.m4a", "gemini-3.8-live", api=api)
# Or select "gemini-3.5-transcribe-live".
print(result["text"])
```

`GeminiAPI.transcribe_live(audio_path, model)` is also available. Existing calls
to `transcribe(..., model="gemini-3.5-transcribe-live", api=api)` or
`api.speech_to_text(audio_input="recording.wav", model="gemini-3.8-live")`
automatically use the Live transport. Existing batch signatures and defaults
are preserved; `gemini-3.5-transcribe` is now an accepted enum selection, and
its `parts[].audioTranscription.text` responses are read alongside ordinary text.

The result has the same keys as batch transcription: `text`, `model`,
`duration`, `processing_time`, `input_tokens`, `output_tokens`, `success`.
Token counts are the latest usage metadata reported by the server, or `None`
when it supplies none. `processing_time` includes streaming at real-time speed.

The client converts to mono 16 kHz PCM16 on temporary local disk, waits for
setup acknowledgement, then sends 100 ms audio chunks between `activityStart`
and `activityEnd`. It receives finalized `inputTranscription` text concurrently,
ignoring interim hypotheses and assistant replies. Gemini 3.8 Live uses the
`AUDIO` response modality; Transcribe Live uses `TEXT`.

Input transcription messages are independently ordered relative to response
completion. After audio ends, Transcribe Live also accepts `generationComplete`
without `turnComplete`; audio-output Live models still require `turnComplete`.
The client drains late input text until one quiet second after completion
(bounded by the configured timeout). This is a best-effort drain;
fragments delayed beyond that interval cannot be guaranteed. An explicitly
unfinished transcript prevents success until completion or timeout.

The key, timeout and base URL come from the supplied `GeminiAPI`. An HTTP(S)
base URL becomes a WS(S) endpoint under the same host and path prefix. There is
no alternate host or model fallback. A shared limiter reserves one request
using the recording's estimated audio-token count. Stream failures, rejected
setup, missing input transcription, server session-ending notices and timeouts
raise `GeminiAPIError`; proactive limiter exceptions propagate unchanged.
The client does not replay a failed session automatically.

This API supports one recording in one connection. It does not implement
session resumption or context compression. Google's documented connection
lifetime is approximately 10 minutes, and an audio-only session without
compression is limited to 15 minutes. These models' request quotas do not mean
unlimited recording duration. `transcribe_large` rejects Live models because
its overlap merger requires REST text generation. Use batch models for that
workflow. Live calls support plain transcription; diarization, analysis,
formatting, prompt and language options through `speech_to_text` are rejected
rather than silently ignored. JSON-mode Live calls return the common result
dictionary without batch `raw_response` data.

Exocortex and Clip Smith can import this function from their installed core.
Their deployment, model selection and application entry points must be
validated in the consumers before claiming application support.

Protocol sources:

- [Google's WebSocket protocol](https://ai.google.dev/api/live)
- [Live transcription and manual VAD](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe)
- [Gemini 3.8 Live modalities](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live)
- [Session management and limits](https://ai.google.dev/gemini-api/docs/live-api/session-management)
