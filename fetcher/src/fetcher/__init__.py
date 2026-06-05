"""fetcher: a local mirror of arxiv papers.

The public SDK lives in ``api.py`` and is re-exported here. The CLI
(``cli.py``) is a thin wrapper over these functions.
"""

from .api import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    classify,
    fetch,
    render_markdown,
    status,
    sync_metadata,
    train_categories,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATA_DIR",
    "classify",
    "fetch",
    "render_markdown",
    "status",
    "sync_metadata",
    "train_categories",
]
