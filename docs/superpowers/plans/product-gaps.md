> Historical design/implementation record. For current setup and structure, see
> [README](../../../README.md) and the [current roadmap](../../roadmap.md).

# Plan: close the docs/PRODUCT.md gaps

Written 2026-10-14 from the gap audit. docs/PRODUCT.md stays owner-edited; nothing here edits it.
Each workstream is test-first (red, then green) and keeps existing public signatures;
behaviour changes are listed in docs/integration.md migration notes.

## Owner decisions (answered 2026-10-14)

| Question in docs/PRODUCT.md | Decision |
|---|---|
| What makes a model "best"; which fallbacks? | Best = first model in `[models] ranking` that is reachable, supports every explicitly requested feature and has quota left. Without `-m`, a failure before output moves to the next ranked model and is recorded. With `-m` there is never a fallback. |
| Provider cannot supply a requested default feature? | Prefer capable models. A missing default-on feature warns on stderr and is recorded in metadata. An explicitly requested non-default value raises `UnsupportedFeatureError`. |
| When is enhancement allowed? | Fully automatic: loudness/noise is assessed on every run; a detected problem triggers enhancement into a separate file beside the source. `--no-enhance` disables it. |
| One model fails during fusion? | Keep every individual output and record each failure. Fuse when at least two succeed; one success is returned labelled "not fused"; none raises. |

## Order (dependency first)

1. Typed failures and one result shape (`errors.py`, None for unknown usage, explicit `no_speech`).
2. One dispatch path (`engine.py`) used by the CLI and the library, for every provider kind.
3. Provider-aware chunking and reconstruction (even chunk sizes, overlap dedupe, surfaced merge failure, speaker relabelling).
4. Speakers / sounds / timestamps options, library arguments mirrored by CLI flags.
5. Selection (probe, capability, quota), fallback, metadata in output, collision-free output names.
6. Reproducible receipts and reviewed-reference scoring.
7. Loudness assessment and automatic enhancement (ffmpeg, already a dependency).
8. Multi-model fusion.
9. Release verifier, benchmark manifest, packaged skill, recovery behaviour, wheel smoke.
