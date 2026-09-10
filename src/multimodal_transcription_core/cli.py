"""Command line boundary for applications that do not import Python packages."""

import argparse
import json
from pathlib import Path
import sys

from .gemini_client import GeminiAPI
from .groq_client import STTAPI, STTModel
from . import http_client


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "http":
        return http_client.main(argv[1:])
    parser = argparse.ArgumentParser(description="Shared transcription core. Use 'http --help' for compatible HTTP servers.")
    parser.add_argument("provider", choices=["gemini", "groq"])
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", required=True, help="Explicit model served by your configured provider")
    parser.add_argument("--language")
    parser.add_argument("--output", required=True, type=Path, help="New JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if not args.audio.is_file():
        parser.error("Audio file does not exist")
    if args.output.exists():
        parser.error("Choose a new output file")
    if not args.output.parent.is_dir():
        parser.error("Create the output directory first")
    if args.provider == "gemini":
        result = GeminiAPI().speech_to_text(str(args.audio), model=args.model, language=args.language)
    else:
        result = STTAPI().speech_to_text(str(args.audio), model=STTModel(args.model), language=args.language, response_format="verbose_json")
    if result.get("success") is False:
        parser.exit(1, "Transcription failed; no successful output written.\n")
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(args.output)
