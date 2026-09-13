# Benchmark

One owner-approved 120-second Russian-language recording, used to compare
transcription models. Each sample lives in its own subfolder so further
recordings can be added without mixing audio or runs.

| Path | Purpose |
|---|---|
| `sample-120s-diary/sample-120s-diary.wav` | the audio |
| `manifest.json` | integrity metadata; `scripts/verify.py` checks the audio against it |
| `reference.template.json` | copy to `reference.local.json` to record review status (stays private) |
| `transcribe.py` | runner that delegates to the installed shared core |

Human references and model outputs stay local; only the audio and this
documentation are public.

## Audio terms

The MIT license covers code and newly written documentation, **not**
`sample-120s-diary/sample-120s-diary.wav`.

That recording is supplied for inspection and reproduction of this benchmark. No
general-purpose data license is granted. Contact the owner for other reuse,
redistribution, training or voice synthesis. Public availability is not consent
to impersonation.

## Producing a human reference

No canonical transcription is supplied — the public status is **unfilled**, and
any machine transcript must remain explicitly unverified.

Listen only to `sample-120s-diary/sample-120s-diary.wav`. Write the spoken
Russian and English words verbatim, including repetitions, false starts and
fillers. Mark unclear spans with a timestamp and `[unclear]`; do not guess
technical names from context. Have a second listener adjudicate uncertain spans.

Save the draft as
`sample-120s-diary/human-transcription-sample-120s-diary.txt` — it is gitignored.
The text file should hold only spoken words; keep timestamps and uncertainty
notes separately. Resolve every `[unclear]` span, or exclude those audio spans
from both reference and hypothesis. Do not set `human_verified` true until the
review has actually happened.

Proposed scoring normalization: Unicode NFC, lowercase, ё to е, punctuation to
spaces, collapsed whitespace. Keep number spelling unchanged; do not expand
acronyms, translate app names, remove fillers or silently rewrite grammar. Agree
this policy, and record the audio hash and annotator process, before calculating
WER/CER. No scores are available yet.
