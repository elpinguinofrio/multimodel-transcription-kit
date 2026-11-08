"""Transcription through the Antigravity CLI (``agy``), using the account it is signed in to.

``agy`` has no audio upload flag, so its agent is asked to read the file with its own
file tool, in plan mode (no commands, no edits), with read access to the file's folder.
The reply is the transcript. Any failure (non-zero exit, timeout, empty reply, a reply
saying it cannot hear audio) raises ProviderError, so the caller moves to its next route.
The binary is $AGY_BIN, else ``agy`` on PATH.
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from .errors import ProviderError
from .transcription_types import is_no_speech

CANNOT_HEAR = "CANNOT_HEAR_AUDIO"
INSTRUCTION = ("Read the audio file {path} with your file-viewing tool and listen to it. Do not run shell "
               "commands. {task} Output only the transcript, with no comments before or after it. If you "
               f"cannot hear the audio, output exactly: {CANNOT_HEAR}")
# agy can end its reply with a clock line of its own, e.g. "[Nov 8 : 4:20pm]"; it is not speech.
_CLOCK_LINE = re.compile(r"(?:^|\n)\s*\[[^\]\n]*\d{1,2}:\d{2}[^\]\n]*\]\s*$")
DEFAULT_TASK = ("Transcribe it verbatim in the language spoken; write English brand names in Latin script.")


class AntigravityError(ProviderError):
    """agy failed, timed out or returned no transcript."""


def binary():
    """Path of the agy CLI, or None when it is not installed."""
    configured = os.environ.get("AGY_BIN")
    if configured:
        return configured if Path(configured).expanduser().is_file() else None
    return shutil.which("agy")


def transcribe(audio, model, *, prompt=None, hint=None, timeout=600, limiter=None):
    """{text, model, provider, ...} for one audio file; raises AntigravityError on failure."""
    agy = binary()
    if agy is None:
        raise AntigravityError("the agy CLI is not installed (set AGY_BIN or put agy on PATH)")
    audio = Path(audio).resolve()
    task = " ".join(p.strip() for p in (prompt or DEFAULT_TASK, hint) if p)
    instruction = INSTRUCTION.format(path=audio, task=task)
    if limiter is not None:
        limiter.acquire(model)
    started = time.perf_counter()
    try:
        done = subprocess.run(
            [agy, "-p", instruction, "--model", model, "--mode", "plan", "--add-dir", str(audio.parent),
             "--print-timeout", f"{int(timeout)}s"],
            cwd=audio.parent, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired as exc:
        raise AntigravityError(f"agy gave no answer within {timeout:.0f}s") from exc
    except OSError as exc:
        raise AntigravityError(f"agy could not start: {exc}") from exc
    text = _CLOCK_LINE.sub("", (done.stdout or "").strip()).strip()
    detail = (done.stderr or "").strip()[-300:]
    if done.returncode != 0:
        raise AntigravityError(f"agy exited {done.returncode}: {detail or 'no message'}")
    if not text:
        raise AntigravityError(f"agy returned no transcript: {detail or 'no message'}")
    if CANNOT_HEAR in text:
        raise AntigravityError("agy could not hear the audio")
    return {"text": text, "model": model, "provider": "agy", "success": True,
            "no_speech": is_no_speech(text), "input_tokens": None, "output_tokens": None,
            "processing_time": time.perf_counter() - started,
            "raw_response": {"stdout": done.stdout, "stderr_tail": detail},
            "request": {"transport": "agy-cli", "model": model, "instruction": instruction}}
