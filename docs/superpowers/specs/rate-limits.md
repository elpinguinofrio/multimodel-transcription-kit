> Historical design/implementation record. For current setup and structure, see
> [README](../../../README.md) and the [current roadmap](../../roadmap.md).

# Spec: respect API limits instead of failing

## Problem
On a limited API tier, transcription fails with "429 too many requests"
instead of waiting. Each app (Exocortex, Clip Smith, ...) would have to
handle this itself.

## Goal
The core handles limits for everyone. A request waits its turn and
then succeeds. Apps don't need any limit logic of their own.

## My limits
Gemini values from AI Studio (Google AI Pro account), read 2026-10-14. Full list:
[gemini-pro-tier-rate-limits.md](gemini-pro-tier-rate-limits.md)

| Model                  | Per minute | Tokens/min | Per day |
|------------------------|------------|------------|---------|
| gemini-3-flash (current default) | 5 | 250K   | 20      |
| gemini-3.5-flash-lite  | 15         | 250K       | 500     |
| gemini-3.1-flash-lite  | 15         | 250K       | 500     |
| gemini-3.5-transcribe  | 3          | 10K        | 25      |
| whisper-large-v3 (Groq) | ?         | ?          | ?       |

## Behaviour
- Before every request: wait if sending now would go over a limit.
- Limits are shared by all apps and runs on this machine, not per call.
- On "429" anyway: wait as long as the server says, then retry.
- Daily limit used up: stop with a clear message saying when it resets
  (don't keep retrying).
- Same behaviour for Gemini and Groq.

## Out of scope
- Switching to another model or provider automatically.
- Paying for or changing tiers.

## Done when
- Tests show that 2 apps sending at once stay under the limit, with no 429.
- One real request on a short generated clip works with my real limits
  (no benchmark runs: another agent owns the benchmark).
