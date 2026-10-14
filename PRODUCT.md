# Product — Multimodal Transcription Core
// human written file, agent's aren't allowed to edit it.

A Python library that turns an audio file into text through a chosen provider(s), and nothing else.

## In scope

- Transcribes one audio file avaliable models; Gemini, Groq models like Whisper, or any OpenAI-compatible endpoint such a local parajeet
- Converts input media to a format the selected provider accepts in memor
- Splits audio over a provider's limit into chunks and merges the results into one transcript.
- cli-runner as an exmaple: chose file, -d destination
- for each model transcription format is this
-- file name (if file no stdou): orignam_file_name.<models>.(attemp if applicable).md
-- in file: src - full path to the source
-- transcription (see samples)
- to make transcription better, can enhance auio files (saves this enhacened files in the folder besides source), reduce noice, isolte voices etc. 
- depends on flag, timestamps are - none, major (like every scene change), by phrases (like a script for reel)
- by default on: transribes and indicate differnet persons who speak,
- by default on: transcirbes non-speak major sounds and effects as in movie subtitles: *sounds of steps*, *barking of dog* etc
- all prev behavior is args that are flags with default on if use cli-runner
- if files saves like in benchmarks, if several run - save all transcritpinos with (1,2,,3) see sample
- root folder have config with all important vars: paths/urls, promts, settings, path to keys, etc. on file


## Out of scope

- Never discovers configuration; every key, model and setting arrives as an argument or an explicit env file.
- Never renders subtitles, edits media, or publishes anything.
- Never contacts a provider at import time.

## Guarantees
- Same file, provider and settings return the same transcript shape.
- Every failure surfaces as a typed provider error, never as a silent empty transcript.
- Timestamps come from provider alignment, never from a language model's guess.

## Consumers
- any repo like Exocortex and Clip-Smith can use it simple way from importing this repo somehow, it would be public on github

## dd
when you run CLI runner and give a path to some file it's in std output, prints the transcript. If no flag provides it use best available model, but also flag can specify the different model basically. You have a test which verifies what each flag works as expected. 
There are 4-unit tests mock output for each provider is an integration test, they should use actual real providers. 
It's expected that not all providers might be available over time, like but the local and host should be available. 