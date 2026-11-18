"""What each piece consumes and produces, and the source to register for it.

The contracts name standard variables in the DYNAMO model catalog. The stored
source is deliberately tiny: it installs this library at a pinned commit and
calls :func:`semantic_sensemaking.pieces.run_piece`, so the code that runs is
the code in this repository at that commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VARIABLE_PREFIX = "https://w3id.org/okn/i/mint/"
REPOSITORY = "helpfultangent/Semantic-Sensemaking"
ARCHIVE_URL = "https://github.com/{repository}/archive/{ref}.tar.gz"

CORPUS = "document_collection__text"
LEXICON = "lexicon__term_set"
BACKBONE = "science_backbone__domain_set"
TERM_COUNTS = "document_lexicon_group__term_count"
ASSIGNMENT = "lexicon_group_science_domain__assignment"
DOMAIN_WEIGHTS = "document_collection_science_domain__weight"
FRAMES = "document_collection_narrative_frame__description"
REPORT = "document_collection__sensemaking_report"


@dataclass(frozen=True)
class PieceSpec:
    name: str
    label: str
    description: str
    inputs: tuple[tuple[str, str], ...]
    outputs: tuple[tuple[str, str], ...]
    # Run arguments the piece accepts, beyond the Tapis token: {argument: description}.
    arguments: dict[str, str] = field(default_factory=dict)
    # The arguments a run must supply. Only these are wired in by default.
    required: tuple[str, ...] = ("output_uri",)
    extras: str = ""

    def env_from_args(self, optional: tuple[str, ...] = ()) -> dict[str, str]:
        """Task input name -> pipeline argument.

        Only the Tapis token and the required arguments are bound by default.
        A workflow engine may refuse a task whose bound argument was not given,
        so an optional setting is bound only when asked for by name.
        """
        unknown = sorted(set(optional) - set(self.arguments))
        if unknown:
            raise ValueError(f"{self.name} has no argument named: {', '.join(unknown)}")
        return {name.upper(): name for name in ("tapis_token", *self.required, *optional)}

    def parameters_schema(self, optional: tuple[str, ...] = ()) -> dict[str, object]:
        """JSON Schema for the run arguments that are bound."""
        names = [*self.required, *optional]
        return {
            "type": "object",
            "properties": {
                name: {"type": "string", "description": self.arguments[name]} for name in names
            },
            "required": list(self.required),
        }


_OUTPUT = {"output_uri": "Folder the piece writes its outputs into (tapis://system/path)."}
_SCORING = {
    "match_mode": "How terms are counted: auto, word_boundary or longest_match.",
    "mapping_method": "How groups are mapped to the backbone: legacy, subdiscipline_aware or embedding.",
    "unmapped_groups": "What to do with groups that map nowhere: drop or keep.",
}
_EMBEDDING = {
    "embedding_kind": "Embedding backend: openai_compatible, huggingface_local or hashing.",
    "embedding_base_url": "Base URL of an OpenAI-compatible embedding server.",
    "embedding_model": "Embedding model name.",
}
_CHAT = {
    "chat_kind": "Chat backend: openai_compatible or anthropic.",
    "chat_base_url": "Base URL of an OpenAI-compatible chat server.",
    "chat_model": "Chat model name.",
}

SPECS: dict[str, PieceSpec] = {
    spec.name: spec
    for spec in (
        PieceSpec(
            "ingest_corpus",
            "Sensemaking: ingest corpus",
            "Read a folder of documents (.txt, .md, .json, .docx, .pdf) into one normalised corpus file.",
            inputs=((CORPUS, "directory"),),
            outputs=((CORPUS, "json"),),
            arguments={
                "source_uri": "Folder of documents to read (tapis://system/path).",
                **_OUTPUT,
                "max_files": "Largest number of files to read.",
                "min_chars": "Skip documents shorter than this many characters.",
            },
            required=("source_uri", "output_uri"),
            extras="readers",
        ),
        PieceSpec(
            "score_lexicon",
            "Sensemaking: score lexicon",
            "Count how often each lexicon group's terms occur in each document.",
            inputs=((CORPUS, "json"), (LEXICON, "json")),
            outputs=((TERM_COUNTS, "csv"),),
            arguments={
                **_OUTPUT,
                "lexicon_uri": "Lexicon to count (JSON or CSV). Default: the built-in lexicon.",
                **_SCORING,
            },
        ),
        PieceSpec(
            "map_and_project",
            "Sensemaking: map and project",
            "Assign lexicon groups to science-backbone domains and project the counts onto them.",
            inputs=((TERM_COUNTS, "csv"), (LEXICON, "json"), (BACKBONE, "json")),
            outputs=((DOMAIN_WEIGHTS, "csv"), (ASSIGNMENT, "csv")),
            arguments={
                **_OUTPUT,
                "taxonomy_uri": "Science backbone (JSON). Default: the built-in backbone.",
                **_SCORING,
            },
        ),
        PieceSpec(
            "discover_frames",
            "Sensemaking: discover frames",
            "Find the recurring frames in a corpus by embedding and clustering its passages, "
            "and export them as a lexicon.",
            inputs=((CORPUS, "json"),),
            outputs=((FRAMES, "csv"), (LEXICON, "json"), (TERM_COUNTS, "csv")),
            arguments={
                **_OUTPUT,
                **_EMBEDDING,
                **_CHAT,
                "max_frames": "Largest number of frames to find.",
                "segment_words": "Words per passage.",
            },
        ),
        PieceSpec(
            "draft_lexicon",
            "Sensemaking: draft lexicon",
            "Draft a lexicon for a problem statement and keep the terms a seed corpus supports. "
            "The result is a draft for a person to review.",
            inputs=((CORPUS, "json"), (BACKBONE, "json")),
            outputs=((LEXICON, "json"), (BACKBONE, "json")),
            arguments={
                **_OUTPUT,
                "problem_statement": "The problem the lexicon should cover.",
                "lexicon_uri": "An existing lexicon to check instead of drafting one.",
                "taxonomy_uri": "Science backbone (JSON). Default: the built-in backbone.",
                "grounding": "How terms are checked against the corpus: keyword, semantic or none.",
                **_CHAT,
            },
        ),
        PieceSpec(
            "render_report",
            "Sensemaking: render report",
            "Write a Markdown summary of the backbone projection, group counts and frames.",
            inputs=((DOMAIN_WEIGHTS, "csv"), (ASSIGNMENT, "csv"), (TERM_COUNTS, "csv")),
            outputs=((REPORT, "md"),),
            arguments={**_OUTPUT},
        ),
    )
}

_SOURCE = '''\
"""Semantic Sensemaking piece: {name} (library commit {ref})."""
import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
    "semantic-sensemaking{extras} @ {archive}",
])


def transform(inputs):
    from semantic_sensemaking.pieces import run_piece

    return run_piece("{name}", inputs)
'''


def _safe(value: str, extra: str) -> bool:
    return bool(value) and all(c.isalnum() or c in extra for c in value)


def stored_source(name: str, ref: str, *, repository: str = REPOSITORY) -> str:
    """The Python source to store with the piece, pinned to library commit ``ref``.

    ``repository`` is the GitHub ``owner/name`` the library is installed from,
    for running a piece from a fork before its commit is upstream.
    """
    spec = SPECS[name]
    if not _safe(ref, "._-"):
        raise ValueError(f"Not a usable git ref: {ref!r}")
    if repository.count("/") != 1 or not _safe(repository, "._-/"):
        raise ValueError(f"Not a GitHub owner/name: {repository!r}")
    return _SOURCE.format(
        name=spec.name,
        ref=ref,
        extras=f"[{spec.extras}]" if spec.extras else "",
        archive=ARCHIVE_URL.format(repository=repository, ref=ref),
    )


def registration(
    name: str, ref: str, *, repository: str = REPOSITORY, optional: tuple[str, ...] = ()
) -> dict[str, object]:
    """Arguments for the dso-dynamo MCP tool ``create_etl_function_piece``.

    ``optional`` names extra settings to wire in besides the required arguments.
    """
    spec = SPECS[name]

    def contracts(pairs: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
        return [{"standard_variable_uri": VARIABLE_PREFIX + v, "format": f} for v, f in pairs]

    return {
        "label": spec.label,
        "description": spec.description,
        "inputs": contracts(spec.inputs),
        "outputs": contracts(spec.outputs),
        "python_source": stored_source(name, ref, repository=repository),
        "entrypoint": "transform",
        "env_from_args": spec.env_from_args(optional),
        "parameters": spec.parameters_schema(optional),
        "version": ref,
    }
