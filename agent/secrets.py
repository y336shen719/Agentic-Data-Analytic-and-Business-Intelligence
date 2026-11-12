"""Read runtime secrets without copying their contents into the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional


class SecretReadError(ValueError):
    """A configured secret file could not provide a nonempty UTF-8 value."""


def read_secret(name: str, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Read NAME_FILE first, otherwise NAME, from the supplied environment.

    A nonblank file setting is authoritative: missing, unreadable, invalid UTF-8,
    or empty files raise SecretReadError instead of falling back to NAME.
    Blank settings are treated as absent. Values have surrounding whitespace
    removed. This function never loads .env, imports Streamlit, or mutates env.
    """
    effective_env = os.environ if env is None else env
    file_setting = (effective_env.get(f"{name}_FILE") or "").strip()
    if file_setting:
        try:
            value = Path(file_setting).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError, ValueError):
            # Do not expose file contents, paths, or underlying exception text.
            raise SecretReadError(f"Unable to read secret file configured by {name}_FILE.") from None
        if not value:
            raise SecretReadError(f"Secret file configured by {name}_FILE is empty.")
        return value

    return (effective_env.get(name) or "").strip() or None
