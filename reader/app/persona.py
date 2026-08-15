"""The reader's persona and custom system prompt.

Both live outside the repo (they're personal): ~/.config/arxiv-reader/,
overridable via READER_PERSONA / READER_SYSTEM_PROMPT. Each is shown in a
chat-pane accordion; the persona is inserted at the top of every new
conversation, the system prompt is appended to every turn's system prompt.
"""

import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "arxiv-reader"


def _read(env_var, filename):
    path = Path(os.environ.get(env_var) or CONFIG_DIR / filename)
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def text():
    return _read("READER_PERSONA", "persona.md")


def system():
    return _read("READER_SYSTEM_PROMPT", "system-prompt.md")
