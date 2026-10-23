Product — Multimodel Transcription Kit

Human-owned. Agents must never edit this file. Suggest changes in chat;
the owner edits it. Generated designs, roadmaps, and plans must preserve
every requirement and flag unresolved decisions.

Features

- Audio file in, text out.
- Providers: Gemini, Groq Whisper, and configured OpenAI-compatible
  endpoints, including local Parakeet.
- Automatically converts input to the provider's format.
- Splits long audio with small overlaps and reconstructs one transcript
  without duplicated or missing boundary speech.
- Shares rate limits and queues across apps to respect provider limits.
- Records statistics for each model: calls, retries, elapsed time, and
  available input/output token usage. Unavailable usage stays unknown.
- Simple CLI: file to stdout, or -d destination directory to save a
  Markdown transcript. Library arguments have corresponding CLI flags
  with consistent defaults. Diagnostics go to stderr.
- Output: <name>.<model>.md, with (2), (3), etc. on reruns. Preserve all
  previous outputs. First line: src: <full path>. Include actual models
  and settings used, such as temperature.
- Picks the best reachable model unless -m specifies one. Model ranking
  and permitted fallback routes must be explicit.
- Speaker labels: on by default, with an option to disable.
- Non-speech sounds, such as *dog barking*: on by default, with an option
  to disable.
- Timestamps: none, major, or phrase; major by default. Times come from
  alignment, never guessed times.
- Automatically checks loudness. Supports optional enhancement for noisy,
  quiet, or loud audio, saved beside the source. Preserve the original
  and record which audio version was transcribed.
- Can run several models on the same input and combine their outputs.
  Preserve individual results and evaluate whether fusion improves
  transcription over the strongest individual model.
- Benchmarks models on approved audio samples. Each run preserves its
  outputs, model names, arguments, settings, and input identity.
  Accuracy scores require a supplied and reviewed human reference.
- Failures raise typed errors, never silent empty transcripts.
  A successful recording with no speech is reported explicitly.
- One explicit root configuration contains paths, URLs, prompts,
  settings, and the key-file path. The library receives configuration
  from its caller; importing it makes no provider calls or data writes.
- Public on GitHub. Never publish private recordings, transcripts,
  references, or credentials.
- Exocortex, Clip Smith, and other apps use the installed library with
  minimal import hassle.
- Claude and Codex can use a skill that invokes transcription through
  this library using the best available model.
- Once finished and tested, it should work a month and a year later
  without human maintenance or routine DevOps. Document external
  dependencies, bounded retries, and recovery behavior.
- Test CLI flags, provider clients, chunk reconstruction, and installed
  package use. Report mocked checks separately from real provider and
  application checks.

Decisions still needed

- What makes a model "best," and which fallback routes are allowed?
- What happens when a provider cannot supply a requested default feature?
- When is automatic enhancement allowed, and what triggers it?
- What happens when one model fails during fusion?