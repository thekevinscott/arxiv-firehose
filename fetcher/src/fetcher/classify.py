"""classify: read each paper's abstract, label with binary topic flags.

For each compiled CoaxedPrompt artifact listed in ``[classify] prompts_dirs``,
fetcher renders the prompt with the paper's metadata, calls the LLM with the
prompt's Pydantic ``response_format`` as a JSON-schema constraint, and
records the typed result. All flags for a paper combine into one
``classification.json`` beside ``metadata.json``.

The LLM backend is an injection seam (``Classifier``); the default talks to a
local Ollama instance. Tests inject a fake without monkeypatching, mirroring
the ``Transport`` / ``Converter`` pattern (see AGENTS.md).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from coaxer import CoaxedPrompt
from pydantic import BaseModel, ValidationError

from .config import Config
from .paths import iter_paper_dirs


class ClassifyError(Exception):
    """The LLM response could not be parsed into the requested schema."""


@dataclass(frozen=True)
class Classifier:
    """Inject seam: ``call(prompt, schema) -> schema-instance``.

    The real implementation hits Ollama; tests inject a fake. fetcher's SDK
    builds the default ``Classifier`` from config at call time.
    """
    call: Callable[[str, type[BaseModel]], BaseModel]


def ollama_classifier(
    model: str,
    *,
    host: str = "http://localhost:11434",
    timeout_s: float = 60.0,
    client: Any = None,
) -> Classifier:
    """Build a Classifier backed by Ollama's structured-output API.

    *client* is the seam unit tests use to inject a fake Ollama. The real
    client is created lazily so importing this module does not require
    ollama to be reachable.
    """
    if client is None:
        from ollama import Client
        client = Client(host=host, timeout=timeout_s)

    def call(prompt: str, schema: type[BaseModel]) -> BaseModel:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema(),
        )
        # The ollama client returns either a dict-like Mapping or a Pydantic
        # ChatResponse object; both expose ``["message"]["content"]``.
        try:
            content = resp["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ClassifyError(f"unexpected ollama response shape: {resp!r}") from exc
        try:
            return schema.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            raise ClassifyError(f"invalid LLM response: {exc}") from exc

    return Classifier(call=call)


def load_coaxed(prompts_dir: Path | str) -> CoaxedPrompt | None:
    """Return the CoaxedPrompt or None if the artifact is missing.

    classify treats a missing or unbuilt prompts dir as "feature not yet
    configured" -- the daily cron must not red-flag while the user is still
    labeling examples and compiling prompts.
    """
    if not prompts_dir:
        return None
    path = Path(prompts_dir)
    if not (path / "prompt.jinja").is_file():
        return None
    return CoaxedPrompt(str(path))


def _flag_name(coaxed: CoaxedPrompt) -> str:
    """The single output field name from a coaxer artifact's response_format.

    coaxer 0.3.x compiles one classifier per labels dir, with exactly one
    output field. Reading the name from ``response_format`` keeps fetcher's
    flag keys in lockstep with whatever the user named ``output_name``
    during ``coax`` compile.
    """
    props = coaxed.response_format.model_json_schema()["properties"]
    return next(iter(props))


def _input_names(coaxed: CoaxedPrompt) -> list[str]:
    """Names of the Jinja input variables the compiled prompt expects."""
    meta_path = Path(coaxed._path) / "meta.json"
    info = json.loads(meta_path.read_text())
    return list(info.get("fields", {}).get("inputs", {}).keys())


def _render_inputs(coaxed: CoaxedPrompt, meta: dict) -> dict[str, str]:
    """Pluck the Jinja inputs the template needs from a paper's metadata.

    Each declared input is stringified; missing inputs default to "" so a
    paper missing a field (e.g. an empty abstract) does not abort the whole
    classify pass.
    """
    return {name: str(meta.get(name, "")) for name in _input_names(coaxed)}


def _write_classification(payload: dict, dest: Path) -> None:
    """Atomically write ``classification.json`` (``.part`` + rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.rename(dest)


def run(
    data_dir: Path,
    config: Config,
    log: logging.Logger,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    classifier: Classifier | None = None,
) -> dict[str, int]:
    """Classify every paper. Returns a counts dict.

    counts: ``{"classified", "cached", "skipped", "failed"}``. When
    ``[classify] prompts_dirs`` is empty the run logs a single "disabled"
    line and returns zeros -- the cron stays green while labels are still
    being authored.
    """
    counts = {"classified": 0, "cached": 0, "skipped": 0, "failed": 0}

    prompts_dirs = list(config.classify.prompts_dirs)
    if not prompts_dirs:
        log.info("classify: disabled (no prompts_dirs configured)")
        return counts

    # Load each CoaxedPrompt once. A missing dir is a warning, not a stop:
    # the user may still be compiling one of several flags.
    coaxed_list: list[tuple[CoaxedPrompt, str]] = []
    for raw in prompts_dirs:
        path = Path(raw)
        cp = load_coaxed(path)
        if cp is None:
            log.warning("classify: missing or unbuilt prompts dir %s", path)
            continue
        coaxed_list.append((cp, _flag_name(cp)))

    if not coaxed_list:
        log.info("classify: no usable prompts dirs")
        return counts

    if classifier is None:
        classifier = ollama_classifier(
            config.classify.model,
            host=config.classify.host,
            timeout_s=config.classify.timeout_s,
        )

    paper_dirs = list(iter_paper_dirs(data_dir))
    log.info("classify start: %d papers, %d classifiers, model=%s",
             len(paper_dirs), len(coaxed_list), config.classify.model)

    processed = 0
    for pd in paper_dirs:
        if limit is not None and processed >= limit:
            break
        try:
            meta = json.loads((pd / "metadata.json").read_text())
            arxiv_id = meta["arxiv_id"]
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            counts["skipped"] += 1
            log.error("skip %s: bad metadata.json", pd.name)
            continue
        processed += 1

        dest = pd / "classification.json"
        if dest.exists() and not force:
            counts["cached"] += 1
            continue

        if dry_run:
            log.info("[dry-run] would classify %s", arxiv_id)
            continue

        try:
            flags: dict[str, Any] = {}
            for coaxed, flag_name in coaxed_list:
                prompt = coaxed(**_render_inputs(coaxed, meta))
                result = classifier.call(prompt, coaxed.response_format)
                flags[flag_name] = getattr(result, flag_name)
            payload = {
                "arxiv_id": arxiv_id,
                "flags": flags,
                "model": config.classify.model,
                "classified_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_classification(payload, dest)
            counts["classified"] += 1
            log.info("class %s: %s", arxiv_id, flags)
        except Exception as exc:  # noqa: BLE001 -- one paper's LLM failure
            counts["failed"] += 1                #     must not abort the run
            log.error("class %s: %s", arxiv_id, exc)

    log.info("classify done: %s", counts)
    return counts
