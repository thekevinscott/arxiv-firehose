"""Ongoing per-paper conversation with Claude via claude-agent-sdk.

Each HTTP request is one turn. Continuity comes from the SDK's session
resume: the first turn's session id is streamed back to the browser, which
sends it with every later turn. Context chips (selected text from the HTML
view, cropped screenshots from the PDF view) arrive as content blocks in
front of the user's question.
"""

import dataclasses

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    query,
)

from . import arxiv, persona

_OPTION_FIELDS = {f.name for f in dataclasses.fields(ClaudeAgentOptions)}


def _options(**kwargs):
    return ClaudeAgentOptions(
        **{k: v for k, v in kwargs.items() if k in _OPTION_FIELDS and v is not None}
    )


def _system_prompt(meta):
    parts = ["You are helping the user read an arxiv paper."]
    if meta:
        authors = ", ".join(meta.get("authors") or [])
        parts[0] = (
            "You are helping the user read an arxiv paper. Ground answers in the "
            "paper; when the user attaches a selection or screenshot, that excerpt "
            "is the context for their question. Be concise and precise.\n\n"
            f"Paper: {meta.get('title')}\n"
            f"arXiv id: {meta.get('arxiv_id')}\n"
            f"Authors: {authors}\n"
            f"Abstract: {meta.get('abstract')}"
        )
    custom = persona.system()
    if custom:
        parts.append("Response style rules (follow strictly):\n\n" + custom)
    return "\n\n".join(parts)


def _seed_text(arxiv_id):
    """First-turn preamble: reader persona, then the whole paper.

    HTML rather than the PDF: the agent-SDK transport has no document
    blocks, and text source lets the model quote the paper verbatim.
    """
    parts = []
    who = persona.text()
    if who:
        parts.append("About me, the reader (calibrate your answers to this):\n\n" + who)
    if arxiv_id and arxiv.valid_id(arxiv_id):
        head = (
            "We are going to discuss the following paper: "
            f"https://arxiv.org/abs/{arxiv_id}\n\n"
        )
        try:
            parts.append(head + "Full HTML source:\n\n" + arxiv.html_for_llm(arxiv_id))
        except Exception:
            parts.append(head + (
                "(This paper has no arxiv HTML version, so the body is not "
                "attached; ground answers in the abstract and the excerpts "
                "the user attaches.)"
            ))
    return "\n\n".join(parts) or None


def _content_blocks(message, context, seed=None, style=None):
    blocks = []
    if seed:
        blocks.append({"type": "text", "text": seed})
    for chip in context or []:
        if chip.get("type") == "text" and chip.get("text"):
            blocks.append(
                {
                    "type": "text",
                    "text": "Selected passage from the paper:\n\n" + chip["text"],
                }
            )
        elif chip.get("type") == "image" and chip.get("data"):
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": chip.get("media_type", "image/png"),
                        "data": chip["data"],
                    },
                }
            )
    blocks.append({"type": "text", "text": message})
    if style:
        # Repeated per turn: with a whole paper in context, system-prompt
        # style rules alone get diluted; proximity to the question wins.
        blocks.append(
            {
                "type": "text",
                "text": "(Standing style rules for this answer — follow them "
                "strictly:\n\n" + style + ")",
            }
        )
    return blocks


async def _input(message, context, seed, style):
    yield {
        "type": "user",
        "message": {
            "role": "user",
            "content": _content_blocks(message, context, seed, style),
        },
    }


async def run(payload, meta, emit):
    """One chat turn. Emits NDJSON events: session, delta, done, error."""
    message = (payload.get("message") or "").strip()
    if not message:
        emit({"type": "error", "message": "empty message"})
        return

    # A fresh conversation (no session to resume) opens with the paper
    # itself; every later turn already has it in the resumed transcript.
    seed = None if payload.get("session_id") else _seed_text(payload.get("arxiv_id"))
    style = persona.system() or None

    options = _options(
        system_prompt=_system_prompt(meta),
        resume=payload.get("session_id") or None,
        model=payload.get("model") or "opus",
        max_turns=1,
        allowed_tools=[],
        disallowed_tools=["Bash", "Read", "Write", "Edit", "WebSearch", "WebFetch", "Task"],
        permission_mode="bypassPermissions",
        setting_sources=[],
        include_partial_messages=True,
    )

    final_parts = []
    async for msg in query(
        prompt=_input(message, payload.get("context"), seed, style), options=options
    ):
        if isinstance(msg, SystemMessage) and msg.subtype == "init":
            session_id = (msg.data or {}).get("session_id")
            if session_id:
                emit({"type": "session", "session_id": session_id})
        elif isinstance(msg, StreamEvent):
            event = msg.event or {}
            if event.get("type") == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    emit({"type": "delta", "text": delta["text"]})
        elif isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text:
                    final_parts.append(block.text)
        elif isinstance(msg, ResultMessage):
            if msg.session_id:
                emit({"type": "session", "session_id": msg.session_id})
    if final_parts:
        # Authoritative copy: the client replaces its delta-accumulated
        # buffer, healing any deltas lost in transit (seen once on iPad).
        emit({"type": "final", "text": "".join(final_parts)})
    emit({"type": "done"})
