# Problem formulation workflow (planned)

**Goal:** help a team turn a problem statement ("How does groundwater policy
interact with coastal subsidence research?") into the two inputs the dictionary
workflow needs — a **lexicon** of concept groups and a **taxonomy** to project
onto — instead of hand-building them in separate notebooks.

This is the inverse of the dictionary workflow: dictionary *reads* a lexicon and
taxonomy; formulation *writes* them. It replaces the SUBSIDE setup and backbone
notebooks that produced `keyword_groups_*.json` and `science_backbone.json`.

## Intended pipeline

| Step | What it does | Shared layer it reuses |
|---|---|---|
| Ingest | Seed corpus (papers, bibliography exports, transcripts) | `corpus.load_corpus` |
| Propose groups | Chat model drafts concept groups and synonym terms for the problem statement, as structured output | `ChatBackend.complete_json` |
| Ground | Keep terms that actually occur (or are semantically present) in the seed corpus | `lexicon.score_corpus`, `lexicon.score_corpus_semantic` |
| Choose backbone | Start from a built-in or supplied taxonomy; optionally propose missing domains | `taxonomy.load_taxonomy` |
| Check coverage | Map groups onto the backbone and flag `Uncategorized` groups for review | `taxonomy.map_lexicon_to_taxonomy` |
| Write | Versioned, schema-valid files for human review | `lexicon.save_lexicon`, `taxonomy.save_taxonomy` |

The output is deliberately a *draft* for people to edit (the files are plain JSON
or CSV) before running the dictionary workflow.

## Configuration

`RunConfig.formulation` (`problem_statement`, `max_groups`, `max_terms_per_group`)
is reserved now; `chat` and `embedding` configure the backends.
