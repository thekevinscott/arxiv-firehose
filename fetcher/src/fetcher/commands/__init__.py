"""One subpackage / module per fetcher command.

A *command* is the unit a CLI invocation maps to. Two cron-level commands
-- ``fetch`` and ``classify`` -- plus ``status`` for read-only counts and
``train-categories`` (developer command) for compiling every labels
subdir into a prompt artifact.

``fetch`` is a subpackage (sync + render stages); ``classify`` is a
subpackage (coaxed prompt loader + HTTP backend + run loop); ``status``
and ``train_categories`` are flat single-file commands. A command is
promoted to a subpackage the moment its internal pieces stop fitting in
one file.
"""
