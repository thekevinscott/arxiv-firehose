"""One subpackage / module per fetcher command.

A *command* is the unit a CLI invocation maps to. ``api.py`` orchestrates
across these (e.g. ``api.run`` runs sync + fetch) but every per-command
loop body lives in here.

Single-file commands (sync, fetch, status) are flat modules. Multi-file
commands (classify -- prompt loading, HTTP backend, store, run) are
subpackages. A command is promoted to a subpackage the moment its
internal pieces stop fitting in one file.
"""
