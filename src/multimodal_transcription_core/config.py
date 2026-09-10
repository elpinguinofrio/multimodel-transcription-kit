"""Applications own their configuration; the core never searches parent folders."""

import os
from pathlib import Path


def load_env_file() -> None:
    """Optionally load an explicit file, preserving existing environment values."""
    selected = os.getenv("TRANSCRIPTION_ENV_FILE")
    if selected:
        path = Path(selected).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"TRANSCRIPTION_ENV_FILE does not exist: {path}")
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
