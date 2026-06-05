"""Cross-command utilities.

Anything imported by more than one command, or by the SDK orchestrator
in ``api.py``: config types, the HTTP transport, the markdown converter,
path conventions, logging setup, the dirsql schema. Not a dumping
ground -- a module belongs here only if it's used outside a single
command.
"""
