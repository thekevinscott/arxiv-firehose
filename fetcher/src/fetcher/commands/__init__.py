"""One subpackage / module per fetcher command.

A *command* is the unit a CLI invocation maps to. The cron-level command
is ``fetch`` (sync metadata, then embed); ``embed`` is also a standalone
command, and ``status`` gives read-only counts.

``fetch`` is a subpackage (sync + embed stages); ``status`` and ``embed``
are flat single-file commands. A command is promoted to a subpackage the
moment its internal pieces stop fitting in one file.
"""
