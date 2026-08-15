"""The reader's persona: who is reading, their knowledge level, familiarity.

Lives outside the repo (it's personal): ~/.config/arxiv-reader/persona.md,
overridable via READER_PERSONA. Shown in the chat pane's accordion and
inserted at the top of every new conversation.
"""

import os
from pathlib import Path

DEFAULT_PATH = Path.home() / ".config" / "arxiv-reader" / "persona.md"


def text():
    path = Path(os.environ.get("READER_PERSONA") or DEFAULT_PATH)
    try:
        return path.read_text().strip()
    except OSError:
        return ""
