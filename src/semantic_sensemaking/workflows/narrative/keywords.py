"""Distinctive terms per cluster (class-based TF-IDF).

These keywords ground each frame in words that actually occur in its passages.
They name frames when no chat model is configured, and they always join the
frame's terms so taxonomy mapping sees corpus vocabulary, not only a model's
paraphrase.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z][a-z\-]+[a-z]")

STOPWORDS = frozenset(
    """
    a about above across after again against all almost also although always am among an and
    another any anyone anything are around as at back be became because become been before being
    below between both but by can cannot could did do does doing done down during each either
    else enough even ever every few for from further get gets getting go going got had has have
    having he her here hers him his how however i if in into is it its itself just know like
    made make many may me might more most much must my need never no nor not now of off often on
    once one only or other others our ours out over own per perhaps quite rather really said same
    say says see seem seems several she should since so some something still such than that the
    their theirs them then there these they thing things think this those though through thus to
    too under until up upon us use used using very was way we well were what when where whether
    which while who whom whose why will with within without would yes yet you your yours
    lot lots kind sort actually right okay ok yeah um uh going gonna want wanted let lets also
    new first two three many much less least
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase content words (3+ letters, internal hyphens kept), stopwords removed."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def cluster_keywords(
    passages: Sequence[str],
    labels: Sequence[int],
    k: int,
    *,
    top_n: int = 8,
) -> list[list[str]]:
    """Return the ``top_n`` most distinctive unigrams/bigrams for each of ``k`` clusters.

    Score = (term count in cluster / tokens in cluster) * log(1 + mean tokens per
    cluster / term count in corpus). A term must occur in at least two of the
    cluster's passages when the cluster has two or more, so one passage's quirk
    does not name a frame. A unigram already covered by a chosen bigram is skipped.
    """
    cluster_counts = [Counter() for _ in range(k)]
    passage_freq = [Counter() for _ in range(k)]
    sizes = Counter(int(c) for c in labels)
    for text, c in zip(passages, labels):
        grams = _grams(_TOKEN.findall(text.lower()))
        cluster_counts[int(c)].update(grams)
        passage_freq[int(c)].update(set(grams))

    corpus_counts: Counter[str] = Counter()
    for counts in cluster_counts:
        corpus_counts.update(counts)
    totals = [sum(c.values()) for c in cluster_counts]
    mean_tokens = (sum(totals) / k) if k else 0.0

    keywords: list[list[str]] = []
    for c in range(k):
        if not totals[c]:
            keywords.append([])
            continue
        min_df = 2 if sizes[c] >= 2 else 1
        scored = sorted(
            (
                (count / totals[c]) * math.log(1 + mean_tokens / corpus_counts[term]),
                term,
            )
            for term, count in cluster_counts[c].items()
            if passage_freq[c][term] >= min_df
        )
        chosen: list[str] = []
        for _score, term in sorted(scored, key=lambda p: (-p[0], p[1])):
            if " " not in term and any(term in other.split() for other in chosen):
                continue
            chosen.append(term)
            if len(chosen) >= top_n:
                break
        keywords.append(chosen)
    return keywords


def _grams(raw_tokens: list[str]) -> list[str]:
    """Content unigrams plus bigrams of *adjacent* content words (no bridging stopwords)."""
    keep = [t not in STOPWORDS for t in raw_tokens]
    unigrams = [t for t, k in zip(raw_tokens, keep) if k]
    bigrams = [
        f"{a} {b}"
        for (a, ka), (b, kb) in zip(zip(raw_tokens, keep), zip(raw_tokens[1:], keep[1:]))
        if ka and kb
    ]
    return unigrams + bigrams
