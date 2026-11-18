"""Turn clusters into named frames, with a chat model or from keywords alone."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from semantic_sensemaking.backends.base import ChatBackend
from semantic_sensemaking.core.types import Lexicon, LexiconEntry
from semantic_sensemaking.errors import BackendError

log = logging.getLogger(__name__)

PASSAGE_WORDS_IN_PROMPT = 150


@dataclass(frozen=True)
class Frame:
    """A recurring way the corpus talks about something, found by clustering passages."""

    frame_id: int
    name: str
    description: str
    terms: tuple[str, ...]
    """Signal terms: the chat model's terms (if any) followed by extracted keywords."""
    keywords: tuple[str, ...]
    """Distinctive words extracted from the frame's passages."""
    passage_ids: tuple[int, ...]
    """Indexes into the passage list, most central first."""
    labeled_by: str
    """``"chat"`` or ``"keywords"``."""

    @property
    def n_passages(self) -> int:
        return len(self.passage_ids)


class FrameLabel(BaseModel):
    """Structured output requested from the chat model for one frame."""

    name: str = Field(description="A short name for the frame, 2-6 words, Title Case")
    description: str = Field(description="One or two sentences: what story or claim this frame carries")
    terms: list[str] = Field(
        min_length=1,
        description="5-10 lowercase words or short phrases that signal this frame in text",
    )


SYSTEM_PROMPT = (
    "You are a qualitative research analyst. You name narrative frames: recurring ways a "
    "body of documents talks about a topic (who acts, what is claimed, what is at stake). "
    "Base every answer only on the passages provided."
)


def keyword_name(frame_id: int, keywords: Sequence[str]) -> str:
    head = " / ".join(keywords[:2]) or "unlabeled"
    return f"F{frame_id + 1}: {head}"


def label_frames(
    clusters: Sequence[tuple[Sequence[int], Sequence[str]]],
    passages: Sequence[str],
    *,
    chat: ChatBackend | None = None,
    representative: int = 3,
) -> list[Frame]:
    """Build one :class:`Frame` per ``(passage_ids, keywords)`` cluster.

    With a chat backend each frame is named from its most central passages; a
    failed or invalid model reply falls back to the keyword name for that frame
    only. Names are made unique so frames can become lexicon groups.
    """
    frames: list[Frame] = []
    used: set[str] = set()
    for frame_id, (ids, keywords) in enumerate(clusters):
        keywords = tuple(keywords)
        name, description, terms, by = keyword_name(frame_id, keywords), "", keywords, "keywords"
        if chat is not None:
            try:
                label = chat.complete_json(
                    _prompt([passages[i] for i in ids[:representative]], keywords),
                    schema=FrameLabel,
                    system=SYSTEM_PROMPT,
                    max_tokens=1024,
                )
                name = label.name.strip() or name
                description = label.description.strip()
                terms = _unique([t.strip().lower() for t in label.terms] + list(keywords))
                by = "chat"
            except (BackendError, ValueError) as exc:
                log.warning("Frame %d: chat labeling failed (%s); using keywords", frame_id + 1, exc)
        if not terms:
            terms = (name,)
        frames.append(
            Frame(
                frame_id=frame_id,
                name=_unique_name(name, used),
                description=description,
                terms=tuple(terms),
                keywords=keywords,
                passage_ids=tuple(int(i) for i in ids),
                labeled_by=by,
            )
        )
    return frames


def frames_to_lexicon(frames: Sequence[Frame], *, name: str = "narrative_frames") -> Lexicon:
    """Frames as a lexicon: reviewable by a person and re-runnable in the dictionary workflow."""
    return Lexicon(
        tuple(LexiconEntry(f.name, f.terms, f.description or None) for f in frames),
        name=name,
        source="narrative workflow",
    )


def _prompt(examples: Sequence[str], keywords: Sequence[str]) -> str:
    blocks = "\n\n".join(
        f"Passage {i + 1}:\n{' '.join(text.split()[:PASSAGE_WORDS_IN_PROMPT])}"
        for i, text in enumerate(examples)
    )
    return (
        "These passages were grouped together because they discuss similar things.\n"
        f"Distinctive words in the group: {', '.join(keywords) or '(none)'}\n\n"
        f"{blocks}\n\n"
        "Name the narrative frame these passages share, describe it, and list terms that "
        "signal it."
    )


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(t for t in items if t))


def _unique_name(name: str, used: set[str]) -> str:
    candidate, n = name, 2
    while candidate in used:
        candidate, n = f"{name} ({n})", n + 1
    used.add(candidate)
    return candidate
