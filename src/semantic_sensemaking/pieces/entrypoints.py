"""The six pieces: each reads its inputs by location, calls the library, and
writes its outputs back.

A piece is called with one dict of string inputs and returns a dict. Input
names are upper-case, as a Tapis Workflows task receives them:

``OUTPUT_URI``      folder the piece writes into (required on the first piece)
``SOURCE_RESULT``   the previous piece's result, as JSON
``<NAME>_URI``      the location of one input, e.g. ``CORPUS_URI``, ``LEXICON_URI``
``TAPIS_TOKEN``     token used for ``tapis://`` locations
``TAPIS_BASE_URL``  Tapis base URL (default https://portals.tapis.io)

Each result carries everything the pieces before it produced plus its own
outputs, so a later piece finds what it needs without being told again. An
explicit ``<NAME>_URI`` input always wins over the carried value.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from semantic_sensemaking._version import __version__
from semantic_sensemaking.backends.factory import get_chat_backend, get_embedding_backend
from semantic_sensemaking.core.config import (
    BackendConfig,
    CorpusConfig,
    FormulationConfig,
    NarrativeConfig,
    ScoringConfig,
)
from semantic_sensemaking.core.types import Corpus, Document, Lexicon, Taxonomy
from semantic_sensemaking.corpus.loader import load_corpus
from semantic_sensemaking.errors import ConfigError
from semantic_sensemaking.lexicon.io import load_lexicon, save_lexicon
from semantic_sensemaking.lexicon.scoring import score_corpus
from semantic_sensemaking.lexicon.semantic import score_corpus_semantic
from semantic_sensemaking.pieces.storage import Storage, join
from semantic_sensemaking.report import markdown_table, render_summary, write_csv
from semantic_sensemaking.taxonomy.io import load_taxonomy, save_taxonomy
from semantic_sensemaking.taxonomy.mapping import map_lexicon_to_taxonomy
from semantic_sensemaking.taxonomy.projection import compare_distributions, project_weights

CORPUS_FILE = "corpus.json"
COUNT_COLUMNS = ["doc_id", "doc", "group", "count"]
_SECRET_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD")


class _Run:
    """One piece invocation: its inputs, what earlier pieces produced, and where it writes."""

    def __init__(self, piece: str, inputs: Mapping[str, Any], storage: Storage | None) -> None:
        self.piece = piece
        self.inputs = {
            str(k).upper(): str(v) for k, v in inputs.items() if v is not None and str(v) != ""
        }
        self.carried = _carried(self.inputs.get("SOURCE_RESULT"))
        self.storage = storage or Storage(
            self.inputs.get("TAPIS_TOKEN"), self.inputs.get("TAPIS_BASE_URL")
        )
        self.output_uri = self.inputs.get("OUTPUT_URI") or self.carried.get("output_uri")
        if not self.output_uri:
            raise ConfigError(f"{piece}: OUTPUT_URI is required (no earlier piece supplied one)")
        self._tmp = tempfile.TemporaryDirectory(prefix=f"ssm-{piece}-")
        self.work = Path(self._tmp.name)
        self.outputs: dict[str, Any] = {}

    def locate(self, name: str) -> str | None:
        return self.inputs.get(f"{name.upper()}_URI") or self.carried.get(f"{name}_uri")

    def fetch(self, name: str, *, required: bool = False) -> Path | None:
        uri = self.locate(name)
        if not uri:
            if required:
                raise ConfigError(f"{self.piece}: {name.upper()}_URI is required")
            return None
        return self.storage.fetch(uri, self.work / "in" / name)

    def setting(self, name: str, default: str | None = None) -> str | None:
        return self.inputs.get(name.upper(), default)

    def local(self, filename: str) -> Path:
        path = self.work / "out" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def publish(self, key: str, path: Path) -> str:
        uri = self.storage.put(path, join(self.output_uri, path.name))
        self.outputs[f"{key}_uri"] = uri
        return uri

    def settings(self) -> dict[str, str]:
        """The inputs this piece was given, without credentials or the chained result."""
        return {
            name: value for name, value in sorted(self.inputs.items())
            if name != "SOURCE_RESULT" and not any(word in name for word in _SECRET_WORDS)
        }

    def result(self, **facts: Any) -> dict[str, Any]:
        record = {"piece": self.piece, "library_version": __version__, "settings": self.settings()}
        path = self.local(f"settings_{self.piece}.json")
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        self.publish(f"settings_{self.piece}", path)
        self._tmp.cleanup()
        return {**self.carried, **self.outputs, **facts,
                "output_uri": self.output_uri, "piece": self.piece}


def _carried(source_result: str | None) -> dict[str, Any]:
    if not source_result:
        return {}
    try:
        value = json.loads(source_result)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


# ── shared readers and writers ──────────────────────────────────────────────


def save_corpus(corpus: Corpus, path: Path) -> Path:
    payload = {
        "documents": [
            {"doc_id": d.doc_id, "label": d.label, "source_kind": d.source_kind, "text": d.text}
            for d in corpus
        ],
        "skipped": [{"path": s.path, "reason": s.reason} for s in corpus.skipped],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def read_corpus(path: Path, config: CorpusConfig | None = None) -> Corpus:
    """A corpus written by :func:`save_corpus`, or a folder of documents."""
    if path.is_file() and path.name == CORPUS_FILE:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Corpus(
            tuple(
                Document(
                    doc_id=d["doc_id"],
                    text=d["text"],
                    label=d.get("label") or d["doc_id"],
                    source_kind=d.get("source_kind", "inline"),
                )
                for d in payload["documents"]
            )
        )
    return load_corpus(path, config)


def _corpus_config(run: _Run) -> CorpusConfig:
    values: dict[str, Any] = {}
    if run.setting("MAX_FILES"):
        values["max_files"] = int(run.setting("MAX_FILES"))
    if run.setting("MIN_CHARS"):
        values["min_chars"] = int(run.setting("MIN_CHARS"))
    if run.setting("RECURSIVE"):
        values["recursive"] = run.setting("RECURSIVE").lower() in {"1", "true", "yes"}
    return CorpusConfig(**values)


def _scoring(run: _Run) -> ScoringConfig:
    values = {
        field: run.setting(name)
        for field, name in (
            ("method", "SCORING_METHOD"),
            ("match_mode", "MATCH_MODE"),
            ("mapping_method", "MAPPING_METHOD"),
            ("unmapped_groups", "UNMAPPED_GROUPS"),
        )
        if run.setting(name)
    }
    return ScoringConfig(**values)


def _backend(run: _Run, prefix: str) -> BackendConfig:
    values = {
        field: run.setting(f"{prefix}_{name}")
        for field, name in (("kind", "KIND"), ("model", "MODEL"), ("base_url", "BASE_URL"))
        if run.setting(f"{prefix}_{name}")
    }
    # The key reaches the backend through the process environment, which is how
    # the library reads keys; it is never written to an output or a result.
    key = run.setting(f"{prefix}_API_KEY") or run.setting("MODEL_API_KEY")
    if key:
        values["api_key_env"] = f"SSM_PIECE_{prefix}_API_KEY"
        os.environ[values["api_key_env"]] = key
    return BackendConfig(**values)


def _lexicon(run: _Run) -> Lexicon:
    path = run.fetch("lexicon")
    return load_lexicon(path) if path else load_lexicon()


def _taxonomy(run: _Run) -> Taxonomy:
    path = run.fetch("taxonomy")
    return load_taxonomy(path) if path else load_taxonomy()


def _counts(frame: pd.DataFrame, *, group: str = "group", count: str = "count") -> pd.DataFrame:
    """One long table shape for term counts and for passages per frame."""
    out = frame.rename(columns={group: "group", count: "count"})
    return out[COUNT_COLUMNS]


def _group_weights(counts: pd.DataFrame, lexicon: Lexicon) -> dict[str, float]:
    totals = {entry.name: 0.0 for entry in lexicon}
    for name, value in counts.groupby("group")["count"].sum().items():
        if name in totals:
            totals[name] = float(value)
    return totals


# ── the pieces ──────────────────────────────────────────────────────────────


def ingest_corpus(inputs: Mapping[str, Any], *, storage: Storage | None = None) -> dict[str, Any]:
    """A folder of documents becomes one normalised corpus file."""
    run = _Run("ingest_corpus", inputs, storage)
    if not run.locate("corpus") and run.inputs.get("SOURCE_URI"):
        run.inputs["CORPUS_URI"] = run.inputs["SOURCE_URI"]
    corpus = read_corpus(run.fetch("corpus", required=True), _corpus_config(run))
    run.publish("corpus", save_corpus(corpus, run.local(CORPUS_FILE)))
    run.publish("corpus_documents", write_csv(corpus.describe(), run.local("corpus_documents.csv")))
    skipped = pd.DataFrame(
        [{"path": s.path, "reason": s.reason} for s in corpus.skipped], columns=["path", "reason"]
    )
    run.publish("skipped_files", write_csv(skipped, run.local("skipped_files.csv")))
    return run.result(documents=len(corpus), skipped_files=len(corpus.skipped))


def score_lexicon(inputs: Mapping[str, Any], *, storage: Storage | None = None) -> dict[str, Any]:
    """Count each lexicon group's terms in each document."""
    run = _Run("score_lexicon", inputs, storage)
    corpus = read_corpus(run.fetch("corpus", required=True), _corpus_config(run))
    lexicon, scoring = _lexicon(run), _scoring(run)
    if scoring.method == "semantic":
        backend = get_embedding_backend(_backend(run, "EMBEDDING"))
        if backend is None:
            raise ConfigError("score_lexicon: SCORING_METHOD=semantic needs EMBEDDING_KIND")
        scores = score_corpus_semantic(
            corpus, lexicon, backend, threshold=scoring.semantic_threshold,
            chunk_size=scoring.chunk_words, chunk_overlap=scoring.chunk_overlap_words,
        )
    else:
        scores = score_corpus(corpus, lexicon, scoring)
    run.publish("term_counts", write_csv(_counts(scores.to_frame()), run.local("term_counts.csv")))
    run.publish("group_totals", write_csv(scores.group_totals(), run.local("group_totals.csv")))
    # Written out so later pieces use the same lexicon, even the built-in one.
    run.publish("lexicon", save_lexicon(lexicon, run.local("lexicon.json")))
    return run.result(total_matches=scores.total_matches(), lexicon_groups=len(lexicon))


def map_and_project(inputs: Mapping[str, Any], *, storage: Storage | None = None) -> dict[str, Any]:
    """Assign lexicon groups to backbone domains and project the counts onto them."""
    run = _Run("map_and_project", inputs, storage)
    lexicon, taxonomy, scoring = _lexicon(run), _taxonomy(run), _scoring(run)
    counts = pd.read_csv(run.fetch("term_counts", required=True), encoding="utf-8-sig")
    backend = None
    if scoring.mapping_method == "embedding":
        backend = get_embedding_backend(_backend(run, "EMBEDDING"))
        if backend is None:
            raise ConfigError("map_and_project: MAPPING_METHOD=embedding needs EMBEDDING_KIND")
    mapping = map_lexicon_to_taxonomy(
        lexicon, taxonomy, top_n=scoring.top_n_assignments,
        method=scoring.mapping_method, backend=backend,
    )
    projection = project_weights(
        _group_weights(counts, lexicon), mapping, unmapped=scoring.unmapped_groups
    )
    run.publish("domain_assignment", write_csv(mapping.to_frame(), run.local("domain_assignment.csv")))
    run.publish("domain_weights", write_csv(projection.domain_totals(), run.local("domain_weights.csv")))
    run.publish(
        "subdiscipline_weights",
        write_csv(projection.subdiscipline_totals(), run.local("subdiscipline_weights.csv")),
    )
    comparison = compare_distributions(mapping.base_distribution(), projection)
    run.publish("comparison", write_csv(comparison, run.local("comparison_base_vs_overlay.csv")))
    run.publish("taxonomy", save_taxonomy(taxonomy, run.local("taxonomy.json")))
    return run.result(total_weight=projection.total_weight(), taxonomy_domains=len(taxonomy))


def discover_frames(inputs: Mapping[str, Any], *, storage: Storage | None = None) -> dict[str, Any]:
    """Find the recurring frames in a corpus and export them as a lexicon."""
    from semantic_sensemaking.workflows.narrative import analyze

    run = _Run("discover_frames", inputs, storage)
    corpus = read_corpus(run.fetch("corpus", required=True), _corpus_config(run))
    embedding = get_embedding_backend(_backend(run, "EMBEDDING"))
    if embedding is None:
        raise ConfigError("discover_frames: EMBEDDING_KIND is required")
    narrative = NarrativeConfig(**{
        field: int(run.setting(name))
        for field, name in (
            ("segment_words", "SEGMENT_WORDS"), ("max_frames", "MAX_FRAMES"),
            ("min_frame_passages", "MIN_FRAME_PASSAGES"), ("seed", "SEED"),
        )
        if run.setting(name)
    })
    result = analyze(
        corpus, _taxonomy(run), embedding, narrative, _scoring(run),
        chat_backend=get_chat_backend(_backend(run, "CHAT")),
    )
    run.publish("frames", write_csv(result.frames_table(), run.local("frames.csv")))
    run.publish("passage_frames", write_csv(result.passages_table(), run.local("passage_frames.csv")))
    run.publish("lexicon", save_lexicon(result.lexicon, run.local("frames_lexicon.json")))
    counts = _counts(result.doc_frame_counts(), group="frame", count="n_passages")
    run.publish("term_counts", write_csv(counts, run.local("term_counts.csv")))
    return run.result(frames=len(result.frames), passages=len(result.passages))


def draft_lexicon(inputs: Mapping[str, Any], *, storage: Storage | None = None) -> dict[str, Any]:
    """Draft a lexicon for a problem statement and keep the terms a seed corpus supports.

    With ``LEXICON_URI`` set, that lexicon is checked against the corpus instead
    and no chat model is called. The result is a draft for a person to review.
    """
    from semantic_sensemaking.workflows.formulation import formulate

    run = _Run("draft_lexicon", inputs, storage)
    corpus_path = run.fetch("corpus")
    corpus = read_corpus(corpus_path, _corpus_config(run)) if corpus_path else None
    existing = run.fetch("lexicon")
    formulation = FormulationConfig(**{
        key: value for key, value in (
            ("problem_statement", run.setting("PROBLEM_STATEMENT")),
            ("grounding", run.setting("GROUNDING")),
            ("propose_domains", (run.setting("PROPOSE_DOMAINS") or "").lower() in {"1", "true", "yes"}),
        ) if value not in (None, "")
    })
    chat = get_chat_backend(_backend(run, "CHAT"))
    if existing is None and chat is None:
        raise ConfigError("draft_lexicon: CHAT_KIND is required unless LEXICON_URI is given")
    scoring = _scoring(run)
    embedding = None
    if formulation.grounding == "semantic" or scoring.mapping_method == "embedding":
        embedding = get_embedding_backend(_backend(run, "EMBEDDING"))
    result = formulate(
        _taxonomy(run), chat, formulation, scoring, corpus=corpus,
        embedding_backend=embedding, proposed=load_lexicon(existing) if existing else None,
    )
    run.publish("lexicon", save_lexicon(result.lexicon, run.local("lexicon_draft.json")))
    run.publish("taxonomy", save_taxonomy(result.taxonomy, run.local("taxonomy_draft.json")))
    for name, table in result.tables().items():
        run.publish(name, write_csv(table, run.local(f"{name}.csv")))
    return run.result(
        review_required=True, lexicon_groups=len(result.lexicon),
        dropped_groups=list(result.dropped_groups), uncovered_groups=result.uncovered_groups(),
    )


def render_report(inputs: Mapping[str, Any], *, storage: Storage | None = None) -> dict[str, Any]:
    """Write a Markdown summary of the tables the earlier pieces produced."""
    run = _Run("render_report", inputs, storage)

    def table(name: str, *, required: bool = False) -> pd.DataFrame | None:
        path = run.fetch(name, required=required)
        return pd.read_csv(path, encoding="utf-8-sig") if path else None

    weights = table("domain_weights", required=True)
    assignment, counts, frames = table("domain_assignment"), table("term_counts"), table("frames")
    facts: list[tuple[str, Any]] = [("Domains with weight", int((weights["weight"] > 0).sum()))]
    sections = [("Science backbone", markdown_table(weights))]
    if counts is not None:
        facts.insert(0, ("Documents", counts["doc_id"].nunique()))
        totals = (
            counts.groupby("group", as_index=False)["count"].sum()
            .sort_values("count", ascending=False, kind="stable")
        )
        sections.append(("Groups", markdown_table(totals, floatfmt=".0f")))
    if frames is not None:
        facts.append(("Frames", len(frames)))
        sections.append(("Frames", markdown_table(frames)))
    if assignment is not None:
        sections.append(("Group to domain", markdown_table(assignment[["group", "primary_domain"]])))
    path = run.local("summary.md")
    path.write_text(render_summary("Semantic sensemaking summary", facts, sections), encoding="utf-8")
    run.publish("report", path)
    return run.result()


PIECES: dict[str, Callable[..., dict[str, Any]]] = {
    "ingest_corpus": ingest_corpus,
    "score_lexicon": score_lexicon,
    "map_and_project": map_and_project,
    "discover_frames": discover_frames,
    "draft_lexicon": draft_lexicon,
    "render_report": render_report,
}


def run_piece(name: str, inputs: Mapping[str, Any], *, storage: Storage | None = None) -> str:
    """Run one piece and return its result as the JSON string a task hands on."""
    try:
        piece = PIECES[name]
    except KeyError:
        raise ConfigError(f"Unknown piece {name!r}; choose from {sorted(PIECES)}") from None
    return json.dumps(piece(inputs, storage=storage), default=str)
