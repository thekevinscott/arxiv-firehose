# fetcher — Agent Guide

Workflow and process for agents working in this repo. Architecture and behavior
live in `README.md`; this file is about *how to work*, not *what the tool does*.

Many conventions here are pulled from
[thekevinscott/dirsql](https://github.com/thekevinscott/dirsql) — when in doubt,
that repo's `AGENTS.md` is the reference.

## Scratch files

Write scratch/temporary files to `/tmp`, not into the repo. Use unique
filenames to avoid collisions with other sessions. Temporary scripts (Python or
shell helpers) also go in `/tmp` and run from there.

## Shell commands

**Do not chain commands** with `;`, `&&`, or `||`. Chaining forces a single
bulk permission decision for the whole pipeline; each command must be evaluated
on its own. Run each as a separate call.

Exceptions: a pipe (`|`) is fine when it is genuinely one operation
(`cmd | jq`). Heredocs are fine. `cd path && cmd` is **not** — pass an absolute
path or `cd` as its own call.

## Testing

### Red/Green development

Follow **red/green** (test-first) methodology:

1. **Write the test first** — it must capture the desired behavior.
2. **Run it and confirm it fails (RED).** Do not proceed until the test is
   reliably red. A test that passes before the implementation exists proves
   nothing.
3. **Make the minimal change to pass (GREEN).** Only then write the
   implementation.
4. **Refactor** if needed, keeping tests green.

### TDD order: outside-in

Write tests before implementation, starting from the outermost layer:

1. **Integration test first** — proves the feature works from the consumer's
   (SDK caller's) perspective.
2. **Unit tests** — written as you implement each module.

A feature is not done until its integration test passes.

### Test layers

- **Unit tests** — colocated with source: `foo.py` → `foo_test.py` in the same
  directory. Test pure functions and small classes in isolation. Dependencies
  are passed as arguments, never patched onto modules.
- **Integration tests** — `tests/integration/`. Exercise the **Python SDK's
  public API** (`fetcher.sync_metadata`, `.fetch`, `.status`, `.run`),
  with third-party I/O (here: all `httpx` network calls) replaced by
  **fixture-backed fakes injected through the public API**. Never test the CLI
  here — the CLI is a thin wrapper over the SDK and has nothing worth testing
  in-process.

### What to mock, what not to

**Monkeypatching is a code smell.** If a test replaces a module-level attribute
on the system under test (`monkeypatch.setattr(download, "httpx", ...)`), the
test is at the wrong layer or the production code is missing a seam.

Concrete rules:

1. **Integration tests hit the SDK's public API**, not the CLI.
2. **Third-party dependencies are injected, not patched.** The network
   transport is a parameter (`transport=`) threaded from the SDK down to the
   downloader. Integration tests pass a fake transport that serves bytes from
   `tests/integration/__fixtures__/`. Production code keeps the real,
   rate-limited httpx transport as the default.
3. **Unit tests** receive their dependencies as arguments or constructor
   params. If a unit can only be tested by patching its imports, refactor for
   dependency injection.

If you reach for `monkeypatch.setattr(some_production_module, ...)`, stop: this
is almost always an integration test that should go through the public API, or
a unit whose target needs an injected parameter.

### pytest-describe

Tests use [`pytest-describe`](https://pypi.org/project/pytest-describe/) BDD
blocks: a `describe_<thing>()` function containing `it_<behaves>()` cases.
Group by the unit under test; name cases for the behavior they assert.

```python
def describe_parse_id():
    def it_parses_modern_ids():
        assert parse_id("2401.12345").yymm == "2401"

    def it_rejects_garbage():
        with pytest.raises(ValueError):
            parse_id("nonsense")
```

### Running tests

`just` is run through uv (the PyPI `just` package is broken; use `rust-just`):

```sh
uvx --from rust-just just              # list recipes
uvx --from rust-just just test-unit    # colocated unit tests, fast
uvx --from rust-just just test-integration
uvx --from rust-just just test         # both
```

If `just` is installed natively, plain `just test` also works.

## SDK / CLI split

Every command is a function in the Python SDK (`fetcher/api.py`, re-exported
from `fetcher/__init__.py`). The CLI (`fetcher/cli.py`) is a thin
typer wrapper: it parses flags and calls the SDK. New behavior goes in the SDK
and gets an integration test there; the CLI only ever grows argument plumbing.

## Package layout

```
src/fetcher/
  api.py            # SDK surface (orchestrator)
  cli.py            # typer wrapper
  commands/         # one subpackage / module per CLI command
    fetch/          # daily ingest: sync metadata + render markdown
      __init__.py   #   the composite (sync → render) lives here
      sync.py       #   stage 1: arxiv RSS -> metadata.json per paper
      render.py     #   stage 2: HTML/LaTeX/PDF -> paper.md per paper
    classify/       # daily classify: abstract -> topic flags via LLM
    status.py       # read-only counts (single-file command)
    train_categories.py  # developer: compile labels/ -> prompts/ (cached)
  shared/           # cross-command utilities (config, paths, logsetup,
                    # download, convert, dirsql_schema)
```

CLI surface mirrors the commands/ folder: `fetcher fetch`, `fetcher
classify`, `fetcher status`, `fetcher train-categories`. The fetch stages
(sync, render) are SDK-only (`api.sync_metadata`, `api.render_markdown`);
they live as separate modules inside `commands/fetch/` so each has its
own integration test file and can be called granularly.

A command is a flat module while it fits in one file (`commands/status.py`,
`commands/train_categories.py`). The moment it needs internal modules
(multiple stages, a backend, a store, a prompt loader) it gets promoted
to a subpackage (`commands/fetch/run.py` + siblings,
`commands/classify/run.py` + siblings). A module belongs in `shared/`
only if it's imported by more than one command or by `api.py` -- not a
dumping ground.
