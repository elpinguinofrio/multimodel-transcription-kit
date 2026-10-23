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


def require_key(name: str, error: type) -> str:
    """Return the named key after loading the explicit env file, or raise ``error``
    naming why it is missing. The message never includes a key value."""
    load_env_file()
    key = os.getenv(name)
    if key:
        return key
    selected = os.getenv("TRANSCRIPTION_ENV_FILE")
    if name in os.environ:
        reason = f"{name} is set but empty in this process's environment"
        if selected:
            reason += f", which hides the value in {selected}; unset {name}"
        else:
            reason += f"; unset {name} or export a real key"
    elif selected:
        reason = f"{Path(selected).expanduser()} has no {name}"
    else:
        reason = (f"{name} is not set and no TRANSCRIPTION_ENV_FILE was given, e.g. "
                  "TRANSCRIPTION_ENV_FILE=~/.config/transcription/.env")
    raise error(f"Set {name} or an explicit TRANSCRIPTION_ENV_FILE: {reason}")
