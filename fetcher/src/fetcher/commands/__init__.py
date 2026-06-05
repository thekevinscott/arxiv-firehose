"""One subpackage / module per fetcher command.

A *command* is the unit a CLI invocation maps to. Two cron-level commands
-- ``fetch`` and ``classify`` -- plus ``status`` for read-only counts.

``fetch`` is a subpackage (sync + render stages); ``classify`` is a
subpackage (coaxed prompt loader + HTTP backend + run loop); ``status``
is a flat single-file command. A command is promoted to a subpackage the
moment its internal pieces stop fitting in one file.
"""
