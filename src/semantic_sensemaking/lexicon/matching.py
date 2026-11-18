"""Count lexicon terms in text."""

from __future__ import annotations

import re
from collections.abc import Sequence

from semantic_sensemaking.core.config import MatchMode

__all__ = ["MatchMode", "TermMatcher", "count_terms"]


class TermMatcher:
    """Precompiled matcher for one group of terms.

    Modes (see :class:`~semantic_sensemaking.core.config.MatchMode`):

    - ``AUTO`` — the prototype's ``count_group_in_text``, reproduced exactly.
      Terms containing a space or hyphen use a plain substring count (so they
      also match inside longer words); other terms use ``\\b`` word boundaries.
      Every term is counted independently, so overlapping terms double count:
      "land subsidence" scores for both ``subsidence`` and ``land subsidence``.
    - ``WORD_BOUNDARY`` — every term, including multi-word ones, must start and
      end on a word boundary; internal whitespace matches any run of whitespace
      (transcripts wrap lines). Overlapping terms still count independently.
    - ``LONGEST_MATCH`` — word boundaries plus a single left-to-right scan that
      prefers the longest term at each position and never reuses matched text,
      so "land subsidence" counts once. Masking is within this group only.
    """

    def __init__(
        self,
        terms: Sequence[str],
        *,
        mode: MatchMode | str = MatchMode.AUTO,
        case_sensitive: bool = False,
    ) -> None:
        self.terms = tuple(terms)
        self.mode = MatchMode(mode)
        self.case_sensitive = case_sensitive
        flags = 0 if case_sensitive else re.IGNORECASE

        if self.mode is MatchMode.AUTO:
            self._auto: list[tuple[str, str, re.Pattern[str] | None]] = []
            for term in self.terms:
                t = term if case_sensitive else term.lower()
                if " " in t or "-" in t:
                    self._auto.append((term, t, None))
                else:
                    self._auto.append((term, t, re.compile(rf"\b{re.escape(t)}\b")))
            return

        self._keys: dict[str, str] = {}
        for term in self.terms:
            self._keys.setdefault(self._normalize(term), term)
        if self.mode is MatchMode.WORD_BOUNDARY:
            self._patterns = {
                key: re.compile(_bounded(key), flags) for key in self._keys
            }
        else:
            ordered = sorted(self._keys, key=len, reverse=True)
            self._alternation = re.compile("|".join(_bounded(k) for k in ordered), flags)

    def _normalize(self, term: str) -> str:
        collapsed = " ".join(term.split())
        return collapsed if self.case_sensitive else collapsed.lower()

    def count(self, text: str) -> int:
        return sum(self.count_detailed(text).values())

    def count_detailed(self, text: str) -> dict[str, int]:
        """Per-term counts keyed by the term as written in the lexicon."""
        if self.mode is MatchMode.AUTO:
            haystack = text if self.case_sensitive else text.lower()
            counts: dict[str, int] = {}
            for term, t, pattern in self._auto:
                n = haystack.count(t) if pattern is None else len(pattern.findall(haystack))
                counts[term] = counts.get(term, 0) + n
            return counts

        counts = {term: 0 for term in self._keys.values()}
        if self.mode is MatchMode.WORD_BOUNDARY:
            for key, pattern in self._patterns.items():
                counts[self._keys[key]] += sum(1 for _ in pattern.finditer(text))
        else:
            for match in self._alternation.finditer(text):
                counts[self._keys[self._normalize(match.group(0))]] += 1
        return counts


def count_terms(
    text: str,
    terms: Sequence[str],
    *,
    mode: MatchMode | str = MatchMode.AUTO,
    case_sensitive: bool = False,
) -> int:
    """Total occurrences of ``terms`` in ``text``. Build a :class:`TermMatcher` to reuse."""
    return TermMatcher(terms, mode=mode, case_sensitive=case_sensitive).count(text)


def _bounded(normalized_term: str) -> str:
    body = r"\s+".join(re.escape(word) for word in normalized_term.split(" "))
    return rf"(?<!\w){body}(?!\w)"
