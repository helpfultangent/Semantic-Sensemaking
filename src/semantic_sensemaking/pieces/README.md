# Pieces

The three workflows, cut into six steps that a workflow engine can run one at a
time. Each piece reads its inputs by location, calls the tool clusters, and
writes its outputs back. They were built to run as Tapis Workflows function
tasks registered in the DYNAMO model catalog, but nothing here depends on that:
a piece is a function that takes a dict and returns a dict.

| Piece | Reads | Writes |
|---|---|---|
| `ingest_corpus` | a folder of documents | `corpus.json`, `corpus_documents.csv`, `skipped_files.csv` |
| `score_lexicon` | corpus, lexicon | `term_counts.csv`, `group_totals.csv`, `lexicon.json` |
| `map_and_project` | term counts, lexicon, backbone | `domain_assignment.csv`, `domain_weights.csv`, `subdiscipline_weights.csv`, `comparison_base_vs_overlay.csv`, `taxonomy.json` |
| `discover_frames` | corpus | `frames.csv`, `passage_frames.csv`, `frames_lexicon.json`, `term_counts.csv` |
| `draft_lexicon` | corpus, backbone, a problem statement | `lexicon_draft.json`, `taxonomy_draft.json`, coverage tables |
| `render_report` | domain weights, assignment, term counts, frames | `summary.md` |

Every piece also writes `settings_<piece>.json`: the library version and the inputs it
was given, without credentials. The pieces write data only. Figures are drawn elsewhere,
from these tables.

A piece that calls a model server takes its key as `MODEL_API_KEY` (or `EMBEDDING_API_KEY` /
`CHAT_API_KEY` when the two servers differ). The key is used for the request only: it is left
out of the settings file, the result and every output.

The flows are chains of these:

- **dictionary**: `ingest_corpus` → `score_lexicon` → `map_and_project` → `render_report`
- **narrative**: `ingest_corpus` → `discover_frames` → `map_and_project` → `render_report`
- **formulation**: `ingest_corpus` → `draft_lexicon`, then a person reviews the draft,
  then the dictionary flow with that lexicon

```python
import json
from semantic_sensemaking import sample_corpus_path
from semantic_sensemaking.pieces import run_piece

result = run_piece("ingest_corpus", {"SOURCE_URI": str(sample_corpus_path()), "OUTPUT_URI": "results/run-1"})
result = run_piece("score_lexicon", {"SOURCE_RESULT": result})
result = run_piece("map_and_project", {"SOURCE_RESULT": result})
print(json.loads(run_piece("render_report", {"SOURCE_RESULT": result}))["report_uri"])
```

## Inputs and results

A piece takes one dict of string inputs, named in upper case as a Tapis
Workflows task receives them, and returns its result as a JSON string.

| Input | Meaning |
|---|---|
| `OUTPUT_URI` | Folder the piece writes into. Required on the first piece of a chain |
| `SOURCE_RESULT` | The previous piece's result |
| `<NAME>_URI` | Location of one input: `CORPUS_URI` (or `SOURCE_URI` for `ingest_corpus`), `LEXICON_URI`, `TAXONOMY_URI`, `TERM_COUNTS_URI`, … |
| `TAPIS_TOKEN`, `TAPIS_BASE_URL` | For `tapis://` locations |

Each result carries everything earlier pieces produced plus the piece's own
outputs (`corpus_uri`, `term_counts_uri`, …), so a later piece finds its inputs
without being told again. An explicit `<NAME>_URI` wins over a carried value.
Lexicon and backbone default to the built-in ones.

Settings are inputs too: `MATCH_MODE`, `MAPPING_METHOD`, `UNMAPPED_GROUPS`,
`SCORING_METHOD`; `EMBEDDING_KIND`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`;
`CHAT_KIND`, `CHAT_BASE_URL`, `CHAT_MODEL`; `MAX_FRAMES`, `SEGMENT_WORDS`;
`PROBLEM_STATEMENT`, `GROUNDING`; `MAX_FILES`, `MIN_CHARS`.

Locations are `tapis://<system>/<path>` (read and written through the Tapis
Files API) or local paths. Term counts and passages-per-frame share one table
shape (`doc_id, doc, group, count`), which is why `map_and_project` serves both
flows.

## Registering a piece

`catalog.py` holds each piece's contract (the standard variables it consumes
and produces) and generates the source to store with it:

```python
from semantic_sensemaking.pieces import registration
args = registration("score_lexicon", "<commit sha>")   # for the dso-dynamo tool create_etl_function_piece
```

The stored source installs this library from GitHub at that commit and calls
`run_piece`, so what runs is this repository's code at that commit. The commit
must be pushed before the piece can run. Pass `repository="owner/name"` to
install from a fork.

A registered piece is wired to its Tapis token and its required arguments only
(`output_uri`, plus `source_uri` for `ingest_corpus`). To expose a setting as a
run argument, name it: `registration("score_lexicon", sha, optional=("lexicon_uri",))`.

Each contract names one input, the one that places the piece in a chain, and
lists the output it hands on first. A planner that chains pieces by matching
output to input would otherwise skip a step: `render_report` also reads term
counts, and declaring that would let it follow `score_lexicon` directly.
`score_lexicon` and `discover_frames` both turn a corpus into term counts, so
such a planner cannot tell the dictionary flow from the narrative flow by
contract alone; register only the pieces of the flow you mean to run, or choose
the piece explicitly.

## Pieces for flows

`flow_catalog.py` describes the same pieces for a DYNAMO flow, which links pieces
as a graph instead of a chain:

- each input arrives as its own file (`SOURCE_0`, `SOURCE_1`, ...), and nothing is
  carried from an earlier piece;
- the piece reports where each of its outputs is, so each can be linked on or
  filed as a result;
- `render_report` comes in two forms, with and without narrative frames, because
  a flow must link every input a piece declares;
- the stored source installs the dependencies in `flow_lock.txt` at exact
  versions with `--require-hashes`, and loads the library from the commit's
  GitHub archive after checking its SHA-256. Nothing is built or resolved at run
  time.

`python -m semantic_sensemaking.pieces.flow_catalog` lists the pieces.
`flow_catalog.registration(slug)` is the catalog row for one.

## Limits

- No piece has been run on Tapis yet. Installing the library inside a function
  task, and the time that takes, are untested.
- `render_report` writes Markdown only. The HTML figures still come from the
  `semantic-sensemaking` command.
- A chat or embedding server that needs an API key is not supported: a function
  task has no way to be given one safely. Use a server reachable without a key.
- `discover_frames` and `draft_lexicon` (when drafting) need a model server the
  task can reach.

## Pieces for DYNAMO flows

`flow_catalog.py` describes each piece as DYNAMO registers it for a flow: its inputs and
outputs, its settings, any credential it needs, and the source DYNAMO stores and runs.
That source is a small wrapper with one function, `transform(inputs)`. It installs exact,
hash-checked dependencies (`flow_lock.txt`) and loads this library from the GitHub archive
of one pinned commit (`REF`), checked against its SHA-256.

The wrappers are committed in `flow_sources/`, one file per piece, so a piece can be
registered from GitHub. On DYNAMO's **Register a piece from GitHub** page, give:

| Field | Value |
|---|---|
| Repository | `helpfultangent/Semantic-Sensemaking` |
| Commit | the full 40-character SHA of a commit on `main` |
| Path to the source file | `src/semantic_sensemaking/pieces/flow_sources/<piece>.py`, e.g. `discover_frames.py` |
| Entrypoint | `transform` |

Copy the inputs, outputs, settings and credential from the piece's entry in `flow_catalog.py`.

After changing the template or a piece, regenerate the files:

```bash
python -m semantic_sensemaking.pieces.flow_catalog --write-sources
```

A test fails if a committed file differs from what `flow_catalog.source()` produces.
