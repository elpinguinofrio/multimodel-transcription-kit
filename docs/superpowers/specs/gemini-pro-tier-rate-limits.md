> Historical design/implementation record. For current setup and structure, see
> [README](../../../README.md) and the [current roadmap](../../roadmap.md).

# Reference: Gemini API rate limits, Google AI Pro account

Account: Google AI Pro subscription. Source: AI Studio "Rate limits by model"
page, copied by the owner on 2026-10-14. Values are limits only (usage dropped).
Stale when: the tier or project changes, or Google changes quotas.
Machine-readable copy used at runtime: `src/multimodel_transcription_kit/tiers/gemini-pro-tier.ini`.
The paste was cut off after "Gemini 3.8 Live Extended Thinking".

| Model | Category | RPM | TPM | RPD |
|---|---|---|---|---|
| Gemini 3.5 Flash Lite | Text-out | 15 | 250K | 500 |
| Gemini 3.1 Flash Lite | Text-out | 15 | 250K | 500 |
| Gemini 3 Flash | Text-out | 5 | 250K | 20 |
| Gemini 3.6 Flash | Text-out | 5 | 250K | 20 |
| Gemini 3.7 Flash | Text-out | 5 | 250K | 20 |
| Gemini 3.8 Flash | Text-out | 5 | 250K | 20 |
| Gemini 2.5 Flash | Text-out | 5 | 250K | 20 |
| Gemini 2.5 Flash Lite | Text-out | 10 | 250K | 20 |
| Gemini 3.5 Transcribe | Live API | 3 | 10K | 25 |
| Gemini 3.5 Transcribe Live | Live API | Unlimited | 20K | Unlimited |
| Gemini 3 Flash Live | Live API | Unlimited | 65K | Unlimited |
| Gemini 3.8 Live | Live API | Unlimited | 65K | Unlimited |
| Gemini 3.5 Live Translate | Live API | Unlimited | 20K | Unlimited |
| Gemini 2.5 Flash Native Audio Dialog | Live API | Unlimited | 1M | Unlimited |
| Gemma 4 26B | Other | 30 | 16K | 14.4K |
| Gemma 4 31B | Other | 30 | 16K | 14.4K |
| Gemini Robotics ER 2 Preview | Other | 5 | 250K | 20 |
| Gemini Embedding 1 | Other | 100 | 30K | 1K |
| Gemini Embedding 2 | Other | 100 | 30K | 1K |
| Antigravity | Agents | 60 | 100K | 100 |
| Gemini 2.5 Flash TTS | Generative | 3 | 10K | 10 |
| Gemini 3.1 Flash TTS | Generative | 3 | 10K | 10 |
| Gemini 3.8 Flash TTS | Generative | 3 | 10K | 10 |
| Gemini 3.8 Flash Lite TTS | Generative | 3 | 10K | 10 |

All limits 0 (not available on this project): Gemini 2 Flash, 2 Flash Lite,
2.5 Pro, 2.5 Pro TTS, 3.1 Pro, 4 Argon, Deep Research Pro Preview, Computer Use
Preview, Omni Flash, Omni 1.1 Flash, all Nano Banana, Lyria 3, Veo 3 models.
