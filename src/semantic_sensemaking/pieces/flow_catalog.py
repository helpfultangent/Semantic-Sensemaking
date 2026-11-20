"""The pieces as DYNAMO registers them for flows.

A flow links pieces as a graph. So here each piece takes its inputs one file
at a time and reports each output's own location, where ``catalog`` describes
pieces that hand one carried result down a chain. The library functions are
the same ones.

The stored source installs exact, hash-checked dependencies (``flow_lock.txt``)
and loads this library from the GitHub archive of one commit, checked against a
recorded SHA-256. ``REF`` is the commit whose code runs; it need not be the
commit this file is in.

Refresh the lock for Python 3.11 on Linux x86_64 with::

    uv pip compile requirements.in --python-version 3.11 \\
        --python-platform x86_64-manylinux_2_28 --generate-hashes --only-binary :all:

keeping, per package, only the hashes of the wheels that platform installs.
"""
from __future__ import annotations

import copy
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REF = "6f586c7f2a81ab9039cdd6d1c13196e1519ef508"
SHORT = REF[:7]
ARCHIVE_SHA256 = "1dcea12d281310e9cd03f1c397fa593a5a0c272796803f341c1b4511fda7777b"
LOCK = open(os.path.join(HERE, "flow_lock.txt"), encoding="utf-8").read()
V = "https://w3id.org/okn/i/mint/"
ID = V + "etl/sensemaking-%s-" + SHORT

CORPUS, LEXICON, BACKBONE = "document_collection__text", "lexicon__term_set", "science_backbone__domain_set"
TERM_COUNTS = "document_lexicon_group__term_count"
ASSIGNMENT = "lexicon_group_science_domain__assignment"
WEIGHTS = "document_collection_science_domain__weight"
FRAMES = "document_collection_narrative_frame__description"
REPORT = "document_collection__sensemaking_report"


def _setting(title, description, type="string", *, default=None, choices=(), minimum=None, maximum=None,
             advanced=False):
    """One setting as a JSON Schema property. ``choices`` is ``[(value, title, description)]``.

    Every default, choice and bound is the library's own: it is copied from the code that
    reads the setting, and a test compares the two, so they cannot drift apart.
    """
    spec = {"title": title, "description": description, "type": type}
    for key, value in (("default", default), ("minimum", minimum), ("maximum", maximum)):
        if value is not None:
            spec[key] = value
    if choices:
        spec["oneOf"] = [{"const": value, "title": name, "description": text} for value, name, text in choices]
    if advanced:
        spec["x-advanced"] = True
    return spec


# Every setting is defined once, so a setting two pieces share is identical in both.
SETTINGS = {
    "output_uri": _setting(
        "Output folder", "Folder the piece writes its outputs into (tapis://system/path). "
        "DYNAMO sets this for a run.", advanced=True),
    "max_files": _setting(
        "Most files to read", "Largest number of files to read from the folder. "
        "Files beyond this are left out.", "integer", default=200, minimum=1, advanced=True),
    "min_chars": _setting(
        "Shortest document", "Skip documents shorter than this many characters.",
        "integer", default=200, minimum=0, advanced=True),
    "lexicon_uri": _setting(
        "Lexicon", "Lexicon to count, as a JSON or CSV file. Leave empty to use the built-in lexicon."),
    "taxonomy_uri": _setting(
        "Science backbone", "Science backbone to map onto, as a JSON file. "
        "Leave empty to use the built-in backbone."),
    "match_mode": _setting(
        "How terms are counted", "How a lexicon term is found in the text.", default="auto", choices=[
            ("auto", "Prototype counting",
             "Terms containing a space or hyphen are counted wherever the text occurs, even inside "
             "longer words. Other terms are counted as whole words. Every term counts on its own, "
             "so 'land subsidence' also counts as 'subsidence'."),
            ("word_boundary", "Whole terms only",
             "Every term, including multi-word ones, must start and end on a word boundary, and "
             "spaces inside a term match any run of whitespace. Terms still count on their own, "
             "so 'land subsidence' also counts as 'subsidence'."),
            ("longest_match", "Longest term wins",
             "Whole terms only, and the longest term at each place wins. Text already counted is "
             "not counted again, so 'land subsidence' counts once and not also as 'subsidence'. "
             "This applies within one lexicon group."),
        ]),
    "mapping_method": _setting(
        "How groups are mapped to the backbone",
        "How each lexicon group is matched to the science backbone's domains and subdisciplines "
        "by comparing words. To compare by meaning, use the piece 'map and project with embeddings'.",
        default="legacy", choices=[
            ("legacy", "Domain score for every subdiscipline",
             "A domain scores one point for each of the group's terms that contains one of the "
             "domain's indicator terms. Every subdiscipline of a matching domain gets that "
             "domain's score, then the three best pairs are kept. Because they tie, that is "
             "usually the first three subdisciplines of one domain."),
            ("subdiscipline_aware", "Top domains, split by subdiscipline",
             "Keeps the three best-scoring domains and splits each domain's score across its "
             "subdisciplines. Subdisciplines whose names overlap the group's terms get a larger "
             "share; with no overlap the score is split evenly."),
        ]),
    "unmapped_groups": _setting(
        "Groups that map nowhere",
        "What happens to a group whose terms match no domain on the backbone.",
        default="drop", advanced=True, choices=[
            ("drop", "Leave out",
             "The group's counts are left out of the domain totals. Its matches do not appear."),
            ("keep", "Keep as Uncategorized",
             "The group's counts stay visible, filed under 'Uncategorized'."),
        ]),
    "embedding_kind": _setting(
        "Embedding model type",
        "Where the embeddings come from. Required by a piece that always uses embeddings; "
        "in 'draft lexicon' only when terms are checked semantically.",
        advanced=True, choices=[
            ("openai_compatible", "Tejas",
             "The model server in your stored 'model_server' credential (an OpenAI-compatible API). "
             "Needs an embedding model name."),
            ("huggingface_local", "Local Hugging Face model",
             "Runs a sentence-transformers model on the machine that runs the piece. Uses "
             "sentence-transformers/all-MiniLM-L6-v2 when no model is named. Needs the optional "
             "sentence-transformers and PyTorch packages, which these flow pieces do not install."),
            ("hashing", "Word hashing (test only)",
             "Offline vectors made by hashing words. Texts that share words are similar. It does "
             "not understand meaning, so use it only for tests and dry runs."),
        ]),
    "embedding_model": _setting(
        "Embedding model name",
        "Name of the embedding model. Required for the model server; the local Hugging Face "
        "model defaults to sentence-transformers/all-MiniLM-L6-v2; word hashing ignores it.",
        advanced=True),
    "chat_kind": _setting(
        "Chat model type",
        "Which service writes text. Required to draft a lexicon. When finding frames, leave it "
        "empty to name each frame from its keywords instead.",
        advanced=True, choices=[
            ("openai_compatible", "Tejas",
             "The model server in your stored 'model_server' credential (an OpenAI-compatible API). "
             "Needs a chat model name."),
            ("anthropic", "Anthropic Claude",
             "Anthropic's Claude, using the key in your 'model_server' credential. Uses claude-opus-5 "
             "when no model is named."),
        ]),
    "chat_model": _setting(
        "Chat model name",
        "Name of the chat model. Required for the model server; Anthropic Claude defaults to "
        "claude-opus-5.", advanced=True),
    "max_frames": _setting(
        "Most frames", "Largest number of frames to find.", "integer", default=12, minimum=1),
    "segment_words": _setting(
        "Words per passage", "Documents are cut into passages of about this many words, and the "
        "passages are grouped into frames.", "integer", default=200, minimum=10, advanced=True),
    "problem_statement": _setting(
        "Problem statement", "The problem the lexicon should cover. Required."),
    "grounding": _setting(
        "How terms are checked against the corpus",
        "Whether the drafted terms must be supported by the seed corpus. Groups left with no "
        "supported term are dropped.", default="keyword", choices=[
            ("keyword", "Term appears in the corpus",
             "A term is kept if it occurs in the corpus at least once, counted the way "
             "'How terms are counted' says. Needs a seed corpus."),
            ("semantic", "Term or similar passage",
             "A term is kept if it occurs in the corpus, or if some passage is similar enough to "
             "it by embedding similarity (at least 0.45). Needs a seed corpus and an embedding "
             "model, so also set the embedding model type and name; it uses the 'model_server' "
             "credential, which must then include the embedding address (its "
             "'embedding_base_url' field)."),
            ("none", "No check",
             "Every drafted term is kept. The seed corpus is not needed."),
        ]),
}

# A model server's key and address are the runner's own, stored once with DYNAMO
# and fetched by the step when it starts. They are never run settings: a value
# typed into a run is kept in many places, and an address chosen by a flow's
# author could send the runner's key elsewhere.
MODEL_SERVER = "model_server"
SECRETS = {
    "discover-frames": [{"name": MODEL_SERVER, "label": "Model server", "env": {
        "MODEL_API_KEY": "api_key", "CHAT_BASE_URL": "chat_base_url",
        "EMBEDDING_BASE_URL": "embedding_base_url"}}],
    # The embedding address is needed only when grounding is semantic.
    "draft-lexicon": [{"name": MODEL_SERVER, "label": "Model server", "env": {
        "MODEL_API_KEY": "api_key", "CHAT_BASE_URL": "chat_base_url",
        "EMBEDDING_BASE_URL": "embedding_base_url"}, "optional": ["embedding_base_url"]}],
    "map-and-project-embedding": [{"name": MODEL_SERVER, "label": "Model server", "env": {
        "MODEL_API_KEY": "api_key", "EMBEDDING_BASE_URL": "embedding_base_url"}}],
}
SENDS_DATA_TO = {
    "discover-frames": [{"service": "The model server in your 'model_server' credential",
                         "what": "Passages of your documents, to embed them and to name the frames found"}],
    "draft-lexicon": [{"service": "The model server in your 'model_server' credential",
                       "what": "Your problem statement and passages of your documents, to draft a lexicon; "
                                "with semantic grounding, also the drafted terms and passages of your documents, "
                                "to embed them"}],
    "map-and-project-embedding": [{"service": "The model server in your 'model_server' credential",
                                   "what": "The lexicon's group names and terms, and the science backbone's "
                                           "domain and subdiscipline names with each domain's indicator terms, "
                                           "to compare them by meaning"}],
}
# Settings a piece fixes, whatever a run says: the piece is the same library function with one
# input decided. Names are the library's input names.
FIXED = {"map-and-project-embedding": {"MAPPING_METHOD": "embedding"}}
# A piece is never changed in place: one whose declaration changed is registered
# under a new id.
REVISION = {
    "ingest-corpus": 2, "score-lexicon": 2, "discover-frames": 3, "map-and-project": 2,
    "draft-lexicon": 4, "render-report": 2, "render-frames-report": 2,
    "map-and-project-embedding": 1,
}

# slug: (library piece, label, description,
#        inputs  [(standard variable, format, the library's input name)],
#        outputs [(standard variable, format, key in the library's result)],
#        settings, the names of the SETTINGS the piece takes)
PIECES = {
    "ingest-corpus": ("ingest_corpus", "Sensemaking: ingest corpus",
        "Read a folder of documents (.txt, .md, .json, .docx, .pdf) into one normalised corpus file.",
        [(CORPUS, "directory", "CORPUS_URI")],
        [(CORPUS, "json", "corpus_uri")],
        ("output_uri", "max_files", "min_chars")),
    "score-lexicon": ("score_lexicon", "Sensemaking: score lexicon",
        "Count how often each lexicon group's terms occur in each document.",
        [(CORPUS, "json", "CORPUS_URI")],
        [(TERM_COUNTS, "csv", "term_counts_uri"), (LEXICON, "json", "lexicon_uri")],
        ("output_uri", "lexicon_uri", "match_mode")),
    "discover-frames": ("discover_frames", "Sensemaking: discover frames",
        "Find the recurring frames in a corpus by embedding and clustering its passages, "
        "and export them as a lexicon.",
        [(CORPUS, "json", "CORPUS_URI")],
        [(TERM_COUNTS, "csv", "term_counts_uri"), (FRAMES, "csv", "frames_uri"), (LEXICON, "json", "lexicon_uri")],
        ("output_uri", "embedding_kind", "embedding_model", "chat_kind", "chat_model", "max_frames",
         "segment_words")),
    "map-and-project": ("map_and_project", "Sensemaking: map and project",
        "Assign lexicon groups to science-backbone domains and project the counts onto them.",
        [(TERM_COUNTS, "csv", "TERM_COUNTS_URI"), (LEXICON, "json", "LEXICON_URI")],
        [(WEIGHTS, "csv", "domain_weights_uri"), (ASSIGNMENT, "csv", "domain_assignment_uri"),
         (BACKBONE, "json", "taxonomy_uri")],
        ("output_uri", "taxonomy_uri", "match_mode", "mapping_method", "unmapped_groups")),
    "map-and-project-embedding": ("map_and_project", "Sensemaking: map and project with embeddings",
        "Assign lexicon groups to science-backbone domains by comparing their meaning with an embedding "
        "model, and project the counts onto them.",
        [(TERM_COUNTS, "csv", "TERM_COUNTS_URI"), (LEXICON, "json", "LEXICON_URI")],
        [(WEIGHTS, "csv", "domain_weights_uri"), (ASSIGNMENT, "csv", "domain_assignment_uri"),
         (BACKBONE, "json", "taxonomy_uri")],
        ("output_uri", "taxonomy_uri", "unmapped_groups", "embedding_kind", "embedding_model")),
    "draft-lexicon": ("draft_lexicon", "Sensemaking: draft lexicon",
        "Draft a lexicon for a problem statement and keep the terms a seed corpus supports. "
        "The result is a draft for a person to review.",
        [(CORPUS, "json", "CORPUS_URI")],
        [(LEXICON, "json", "lexicon_uri"), (BACKBONE, "json", "taxonomy_uri")],
        ("output_uri", "problem_statement", "grounding", "embedding_kind", "embedding_model",
         "chat_kind", "chat_model")),
    "render-report": ("render_report", "Sensemaking: render report",
        "Write a Markdown summary of the backbone projection and group counts.",
        [(WEIGHTS, "csv", "DOMAIN_WEIGHTS_URI"), (ASSIGNMENT, "csv", "DOMAIN_ASSIGNMENT_URI"),
         (TERM_COUNTS, "csv", "TERM_COUNTS_URI")],
        [(REPORT, "md", "report_uri")],
        ("output_uri",)),
    "render-frames-report": ("render_report", "Sensemaking: render report with frames",
        "Write a Markdown summary of the backbone projection, group counts and narrative frames.",
        [(WEIGHTS, "csv", "DOMAIN_WEIGHTS_URI"), (ASSIGNMENT, "csv", "DOMAIN_ASSIGNMENT_URI"),
         (TERM_COUNTS, "csv", "TERM_COUNTS_URI"), (FRAMES, "csv", "FRAMES_URI")],
        [(REPORT, "md", "report_uri")],
        ("output_uri",)),
}

SOURCE = '''"""Semantic Sensemaking piece: __LABEL__.

Runs __PIECE__ from helpfultangent/Semantic-Sensemaking at commit
__REF__.
Its inputs arrive one file each (SOURCE_<position>) and it reports where each
of its outputs is. Dependencies are installed at exact versions, each checked
against its hash; the library is loaded from the commit's archive, checked
against a recorded SHA-256. Nothing else is installed.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

_PIECE = "__PIECE__"
_REF = "__REF__"
_ARCHIVE = "https://github.com/helpfultangent/Semantic-Sensemaking/archive/" + _REF + ".tar.gz"
_ARCHIVE_SHA256 = "__ARCHIVE_SHA256__"
# input position -> the library's name for that input
_INPUTS = __INPUTS__
# output position -> where the library's result says that output is
_OUTPUTS = __OUTPUTS__
__FIXED_DEF___LOCK = """\\
__LOCK__"""


def _load_library():
    work = tempfile.mkdtemp(prefix="ssm-")
    lock = os.path.join(work, "lock.txt")
    with open(lock, "w", encoding="utf-8") as handle:
        handle.write(_LOCK)
    subprocess.check_call([
        sys.executable, "-m", "pip", "--isolated", "install", "--quiet", "--disable-pip-version-check",
        "--no-input", "--require-hashes", "--no-deps", "--only-binary=:all:",
        "--index-url", "https://pypi.org/simple", "-r", lock,
    ])
    with urllib.request.urlopen(_ARCHIVE, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != _ARCHIVE_SHA256:
        raise RuntimeError("the library archive is not the one this piece was approved with")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(work, filter="data")
        else:  # an older Python: the archive's hash was checked above, and nothing odd is let through
            for member in archive.getmembers():
                if member.name.startswith("/") or ".." in member.name.split("/") or member.issym() or member.islnk():
                    raise RuntimeError("the library archive holds a path this piece will not extract")
            archive.extractall(work)
    sys.path.insert(0, os.path.join(work, "Semantic-Sensemaking-" + _REF, "src"))


def transform(inputs):
    given = {str(k): v for k, v in (inputs or {}).items() if v is not None and str(v) != ""}
    # Each input is linked by itself; nothing is carried over from an earlier piece.
    given.pop("SOURCE_RESULT", None)
__FIXED_USE__    for position, name in _INPUTS.items():
        value = given.pop("SOURCE_" + position, None)
        if value:
            given[name] = value
    missing = sorted(name for name in _INPUTS.values() if name not in given)
    if missing:
        raise ValueError(_PIECE + ": not linked to an input: " + ", ".join(missing))
    _load_library()
    from semantic_sensemaking.pieces import run_piece

    result = json.loads(run_piece(_PIECE, given))
    return {"svo_flow": {"outputs": {position: result[key] for position, key in _OUTPUTS.items()}}}
'''


def _fixed_def(slug: str) -> str:
    """Nothing for a piece that fixes nothing, so its source is unchanged."""
    return f"# settings this piece fixes, whatever a run says\n_FIXED = {FIXED[slug]!r}\n" if slug in FIXED else ""


def _fixed_use(slug: str) -> str:
    return "    given.update(_FIXED)\n" if slug in FIXED else ""


def source(slug: str) -> str:
    piece, label, _, inputs, outputs, _ = PIECES[slug]
    return (SOURCE.replace("__LABEL__", label).replace("__PIECE__", piece).replace("__REF__", REF)
            .replace("__ARCHIVE_SHA256__", ARCHIVE_SHA256)
            .replace("__INPUTS__", repr({str(i): name for i, (_, _, name) in enumerate(inputs)}))
            .replace("__OUTPUTS__", repr({str(i): key for i, (_, _, key) in enumerate(outputs)}))
            .replace("__FIXED_DEF__", _fixed_def(slug)).replace("__FIXED_USE__", _fixed_use(slug))
            .replace("__LOCK__", LOCK))


def piece_id(slug: str) -> str:
    revision = REVISION.get(slug)
    return ID % slug + (f"-r{revision}" if revision else "")


def registration(slug: str) -> dict:
    """The catalog row for one piece: insert_modelcatalog_etl_process_one's object."""
    _, label, description, inputs, outputs, arguments = PIECES[slug]
    contracts = [{"role": role, "position": position, "standard_variable_uri": V + variable, "format": fmt}
                 for role, items in (("input", inputs), ("output", outputs))
                 for position, (variable, fmt, _) in enumerate(items)]
    names = sorted(arguments)
    return {
        "id": piece_id(slug), "label": label, "description": description, "version": SHORT,
        "visibility": "public", "runtime_kind": "tapis_function",
        "runtime_json": {
            "language": "python", "entrypoint": "transform",
            "source": {"language": "python", "code": source(slug)},
            "env_from_args": {"TAPIS_TOKEN": "tapis_token", **{name.upper(): name for name in names}},
            "parameters_schema_json": {
                "type": "object",
                "properties": {name: copy.deepcopy(SETTINGS[name]) for name in names},
                "required": ["output_uri"],
            },
            **({"secrets": SECRETS[slug]} if slug in SECRETS else {}),
            **({"sends_data_to": SENDS_DATA_TO[slug]} if slug in SENDS_DATA_TO else {}),
        },
        "contracts": {"data": contracts},
    }


SOURCES_DIR = os.path.join(HERE, "flow_sources")


def source_path(slug: str) -> str:
    """Where a piece's source is committed, so DYNAMO can register it from GitHub:
    ``src/semantic_sensemaking/pieces/flow_sources/<slug with underscores>.py``."""
    return os.path.join(SOURCES_DIR, slug.replace("-", "_") + ".py")


def write_sources() -> list[str]:
    """Write every piece's source to its committed file. A test fails if a committed
    file differs from ``source(slug)``, so rerun this after changing the template."""
    os.makedirs(SOURCES_DIR, exist_ok=True)
    written = []
    for slug in PIECES:
        with open(source_path(slug), "w", encoding="utf-8") as handle:
            handle.write(source(slug))
        written.append(source_path(slug))
    return written


if __name__ == "__main__":
    import sys

    if "--write-sources" in sys.argv:
        for path in write_sources():
            print("wrote", os.path.relpath(path))
        raise SystemExit(0)
    for slug in PIECES:
        row = registration(slug)
        compile(source(slug), slug, "exec")
        print(row["id"].rsplit("/", 1)[-1], "| source", len(source(slug)), "bytes | contracts",
              [(c["role"][0], c["position"], c["standard_variable_uri"].rsplit("/", 1)[-1][:28], c["format"]) for c in row["contracts"]["data"]])
