# Human reference instructions

Listen only to `sample-120s-diary/sample-120s-diary.wav`. Write the spoken Russian and English words verbatim, including repetitions, false starts and fillers. Mark unclear spans with a timestamp and `[unclear]`. Do not guess technical names from context.

Have a second listener adjudicate uncertain spans. Record the audio hash, annotator process, and normalization decisions before calculating WER/CER. Evaluate hypotheses generated from this exact 120-second input.

Public status: **unfilled; no canonical transcription is supplied**. A machine transcript must remain explicitly unverified.

Save your draft as
`benchmark/sample-120s-diary/human-transcription-sample-120s-diary.txt`, or send it
to your agent. That local draft is ignored by Git. Copy `reference.template.json`
to `reference.local.json` when recording review status; both local reference files
stay private until explicitly approved for publication. Do not mark
`human_verified` true until the human review has happened.

The text file should contain only spoken words. Keep timestamps and uncertainty
notes separately before scoring. Resolve every `[unclear]` span, or explicitly
exclude those audio spans from both reference and hypothesis. Preserve fillers,
repetitions, false starts and Russian/English spelling.

Proposed scoring normalization: Unicode NFC, lowercase, ё to е, punctuation to
spaces, collapsed whitespace. Keep number spelling unchanged; do not expand
acronyms, translate app names, remove fillers or silently rewrite grammar. Agree
this policy before calculating WER/CER. No scores are available yet.
