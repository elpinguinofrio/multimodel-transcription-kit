# Roadmap

[Product specification](PRODUCT.md) remains the owner-controlled specification. This is
an implementation map, not a claim that every product requirement is validated.
It consolidates the two October 14 roadmap drafts; proposed versions 0.4–1.0
were never agreed release commitments. Package metadata currently declares 0.3.0.

## Existing implementation

- One library engine with Gemini REST/Live, Groq, and OpenAI-compatible clients.
- CLI flags for speakers, sounds, timestamps, prompts, enhancement, and fusion.
- Configured model ranking/fallback, shared limits, and request usage history.
- Audio conversion, overlapping chunks, reconstruction, typed failures, and silence handling.
- Numbered outputs, benchmark receipts/replay, and reviewed-reference scoring.
- Agent skill, private host launcher, offline tests, and a Git-index release check.

See [integration](integration.md) for public APIs and migration notes,
[operations](operations.md) for dependencies/recovery, and
[Live transcription](live-transcription.md) for transport limits.
The CLI and benchmark use the root configuration; library callers supply values
explicitly. There is no separate quickstart application or publication inventory.

## Remaining acceptance work

- Validate Exocortex through its installed core and real application entry point
  before removing its existing implementation; validate Clip Smith separately.
- Repeat real-provider checks for the selected models and routes, including Live
  input completion and batch audio-transcription responses. Offline fixtures do
  not establish service availability or transcript quality.
- Complete the approved benchmark comparisons and any fusion-quality evaluation.
  Accuracy claims require a reviewed human-from-audio reference; keep reference
  drafts and machine transcripts private.
- Validate long recordings and installed-wheel behavior on each consuming machine.
  Host and VM paths/runtimes and local SQLite state must be checked independently.
- Resolve the enhancement-location discrepancy: docs/PRODUCT.md says saved beside the
  source; current code/migration notes use temporary copies unless explicitly kept.
  This cleanup does not change that behavior or the owner's specification.
- Keep model ranking, fallback routes, unsupported-feature policy, and fusion
  failure policy explicit. Recorded owner decisions are in the historical
  [product-gaps plan](superpowers/plans/product-gaps.md); docs/PRODUCT.md still lists
  those decisions as open.

## Deferred proposals from the original drafts

Prompt/language controls, output features, enhancement, model expiry checks,
provider tests, and consumer migration were originally proposed as versions
0.4–1.0. Several now have implementations above; acceptance still requires the
relevant execution evidence. Optional weekly live smoke checks would require
separate scheduling authorization and are not installed. Publishing/tagging a
version for consumer pinning also requires a separate instruction.

The original drafts cited private benchmark investigation notes and tentative
prompt comparisons. Those do not establish public accuracy or benchmark completion.

## Original owner-written draft

Preserved verbatim from the earlier roadmap; docs/PRODUCT.md is the canonical spec.

> **HUMAN-SPEC** · Owner-written. Read-only for agents: do not edit anything until HUMAN-SPEC-END.

## Features

- Audio file in, text out.
- Providers: Gemini, Groq Whisper, any OpenAI-compatible endpoint (local Parakeet).
- auto Converts input to the provider's format.
- Splits long audio, merges the chunks, samrtley: with little overlaps, and reconsutct trancript properly
- Shares rate limits across apps, and queue to respect provider's rpm and other limits
- calc staticstic for each model: n of calls, tookens in/out etc
- simple-cli: file → stdout, or `-d` for desitation to creat .md file with transcript. clid accept all args that tranciber provides with deafult values if no flag privdede, same as for args.
-  Output: `<name>.<model>.md`, `(2)`, `(3)` on reruns, fist like `src:` <full path>, model: <defialt model and settings like temperature>
  args/flags:
- Picks the best reachable model; unless `-m`  specifies the model 
- Speaker labels: on by default.
- Non-speech sounds (`*dog barking*`): on by default.
- Timestamps: none | major | phrase. (major by default)
- check auto loudness, does option audio enhancement if too noice/quite/loud, saved beside the source.
- can speficy that run several models on same input in kinda fusion way, that smarterly assemples the best possible transcripino using this several models's output
- 
benchmarks for differnt model for existing providede audio samples
each run create files with model name, args,

- Failures raise typed errors, never empty text.
- project wil lbe public on GitHub;
- project/model used by Exocortex and Clip Smith and other with min import hastle.
- claude/codex can use this module as a skill that call transcibe that uses best avalieble model

- this module onces fnished and tested, should work in 1m, 1year from no with NO HUMAN maintance. just work, doesn't requite lots of devops ideally not any at all

> **HUMAN-SPEC-END**
