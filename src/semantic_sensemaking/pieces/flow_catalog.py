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

OUT = {"output_uri": "Folder the piece writes its outputs into (tapis://system/path)."}
SCORING = {
    "match_mode": "How terms are counted: auto, word_boundary or longest_match.",
    "mapping_method": "How groups are mapped to the backbone: legacy, subdiscipline_aware or embedding.",
    "unmapped_groups": "What to do with groups that map nowhere: drop or keep.",
}
EMBEDDING = {
    "embedding_kind": "Embedding backend: openai_compatible, huggingface_local or hashing.",
    "embedding_model": "Embedding model name.",
}
CHAT = {
    "chat_kind": "Chat backend: openai_compatible or anthropic.",
    "chat_model": "Chat model name.",
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
    "draft-lexicon": [{"name": MODEL_SERVER, "label": "Model server", "env": {
        "MODEL_API_KEY": "api_key", "CHAT_BASE_URL": "chat_base_url"}}],
}
SENDS_DATA_TO = {
    "discover-frames": [{"service": "The model server in your 'model_server' credential",
                         "what": "Passages of your documents, to embed them and to name the frames found"}],
    "draft-lexicon": [{"service": "The model server in your 'model_server' credential",
                       "what": "Your problem statement and passages of your documents, to draft a lexicon"}],
}
# A piece is never changed in place: one whose declaration changed is registered
# under a new id.
REVISION = {"discover-frames": 2, "draft-lexicon": 2}

# slug: (library piece, label, description,
#        inputs  [(standard variable, format, the library's input name)],
#        outputs [(standard variable, format, key in the library's result)],
#        arguments {name: description})
PIECES = {
    "ingest-corpus": ("ingest_corpus", "Sensemaking: ingest corpus",
        "Read a folder of documents (.txt, .md, .json, .docx, .pdf) into one normalised corpus file.",
        [(CORPUS, "directory", "CORPUS_URI")],
        [(CORPUS, "json", "corpus_uri")],
        {**OUT, "max_files": "Largest number of files to read.",
         "min_chars": "Skip documents shorter than this many characters."}),
    "score-lexicon": ("score_lexicon", "Sensemaking: score lexicon",
        "Count how often each lexicon group's terms occur in each document.",
        [(CORPUS, "json", "CORPUS_URI")],
        [(TERM_COUNTS, "csv", "term_counts_uri"), (LEXICON, "json", "lexicon_uri")],
        {**OUT, "lexicon_uri": "Lexicon to count (JSON or CSV). Default: the built-in lexicon.",
         "match_mode": SCORING["match_mode"]}),
    "discover-frames": ("discover_frames", "Sensemaking: discover frames",
        "Find the recurring frames in a corpus by embedding and clustering its passages, "
        "and export them as a lexicon.",
        [(CORPUS, "json", "CORPUS_URI")],
        [(TERM_COUNTS, "csv", "term_counts_uri"), (FRAMES, "csv", "frames_uri"), (LEXICON, "json", "lexicon_uri")],
        {**OUT, **EMBEDDING, **CHAT, "max_frames": "Largest number of frames to find.",
         "segment_words": "Words per passage."}),
    "map-and-project": ("map_and_project", "Sensemaking: map and project",
        "Assign lexicon groups to science-backbone domains and project the counts onto them.",
        [(TERM_COUNTS, "csv", "TERM_COUNTS_URI"), (LEXICON, "json", "LEXICON_URI")],
        [(WEIGHTS, "csv", "domain_weights_uri"), (ASSIGNMENT, "csv", "domain_assignment_uri"),
         (BACKBONE, "json", "taxonomy_uri")],
        {**OUT, "taxonomy_uri": "Science backbone (JSON). Default: the built-in backbone.", **SCORING}),
    "draft-lexicon": ("draft_lexicon", "Sensemaking: draft lexicon",
        "Draft a lexicon for a problem statement and keep the terms a seed corpus supports. "
        "The result is a draft for a person to review.",
        [(CORPUS, "json", "CORPUS_URI")],
        [(LEXICON, "json", "lexicon_uri"), (BACKBONE, "json", "taxonomy_uri")],
        {**OUT, "problem_statement": "The problem the lexicon should cover.",
         "grounding": "How terms are checked against the corpus: keyword, semantic or none.", **CHAT}),
    "render-report": ("render_report", "Sensemaking: render report",
        "Write a Markdown summary of the backbone projection and group counts.",
        [(WEIGHTS, "csv", "DOMAIN_WEIGHTS_URI"), (ASSIGNMENT, "csv", "DOMAIN_ASSIGNMENT_URI"),
         (TERM_COUNTS, "csv", "TERM_COUNTS_URI")],
        [(REPORT, "md", "report_uri")],
        {**OUT}),
    "render-frames-report": ("render_report", "Sensemaking: render report with frames",
        "Write a Markdown summary of the backbone projection, group counts and narrative frames.",
        [(WEIGHTS, "csv", "DOMAIN_WEIGHTS_URI"), (ASSIGNMENT, "csv", "DOMAIN_ASSIGNMENT_URI"),
         (TERM_COUNTS, "csv", "TERM_COUNTS_URI"), (FRAMES, "csv", "FRAMES_URI")],
        [(REPORT, "md", "report_uri")],
        {**OUT}),
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
_LOCK = """\\
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
    for position, name in _INPUTS.items():
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


def source(slug: str) -> str:
    piece, label, _, inputs, outputs, _ = PIECES[slug]
    return (SOURCE.replace("__LABEL__", label).replace("__PIECE__", piece).replace("__REF__", REF)
            .replace("__ARCHIVE_SHA256__", ARCHIVE_SHA256)
            .replace("__INPUTS__", repr({str(i): name for i, (_, _, name) in enumerate(inputs)}))
            .replace("__OUTPUTS__", repr({str(i): key for i, (_, _, key) in enumerate(outputs)}))
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
                "properties": {name: {"type": "string", "description": arguments[name]} for name in names},
                "required": ["output_uri"],
            },
            **({"secrets": SECRETS[slug]} if slug in SECRETS else {}),
            **({"sends_data_to": SENDS_DATA_TO[slug]} if slug in SENDS_DATA_TO else {}),
        },
        "contracts": {"data": contracts},
    }


if __name__ == "__main__":
    for slug in PIECES:
        row = registration(slug)
        compile(source(slug), slug, "exec")
        print(row["id"].rsplit("/", 1)[-1], "| source", len(source(slug)), "bytes | contracts",
              [(c["role"][0], c["position"], c["standard_variable_uri"].rsplit("/", 1)[-1][:28], c["format"]) for c in row["contracts"]["data"]])
