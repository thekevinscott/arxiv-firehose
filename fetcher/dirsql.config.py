"""dirsql config entry point at the fetcher repo root.

Logic lives in ``fetcher.dirsql_schema`` so production code can import it
without ``importlib.util.spec_from_file_location`` gymnastics. This file
stays at repo root so that, once dirsql PR #220 (native-language config
files) lands, the dirsql CLI can discover it by convention.

Until then, fetcher uses ``build_app`` directly in-process (see
``classify.run`` and the feed generator).
"""
from fetcher.dirsql_schema import build_app

__all__ = ["build_app"]
