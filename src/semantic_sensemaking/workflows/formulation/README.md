# Problem formulation workflow

**Goal:** help a team turn a problem statement ("How does groundwater policy
interact with coastal subsidence research?") into the two inputs the dictionary
workflow needs — a **lexicon** of concept groups and a **taxonomy** to project
onto — instead of hand-building them in separate notebooks.

This is the inverse of the dictionary workflow: dictionary *reads* a lexicon and
taxonomy; formulation *writes* them. It replaces the SUBSIDE setup and backbone
notebooks that produced `keyword_groups_*.json` and `science_backbone.json`.

```bash
semantic-sensemaking formulation --config configs/formulation-example.yaml --corpus path/to/seed_docs
# offline, no chat model: check an existing lexicon against a corpus
semantic-sensemaking formulation --sample-corpus --lexicon my_lexicon.csv
```

## Pipeline

| Step | What it does | Code |
|---|---|---|
| Propose groups | Chat model drafts concept groups and their terms for `formulation.problem_statement`, as structured output, with the backbone's domain names as context. Skipped when `inputs.lexicon` is given — the existing lexicon is checked instead | `propose.propose_lexicon` → `ChatBackend.complete_json` |
| Ground | Keep terms that occur at least `min_term_count` times in the seed corpus (`grounding: keyword`), or that also match a passage by meaning (`grounding: semantic`). Groups left with fewer than `min_terms_per_group` terms are dropped | `grounding.py` → `lexicon.TermMatcher` |
| Choose backbone | A built-in or supplied taxonomy | `taxonomy.load_taxonomy` |
| Check coverage | Map surviving groups onto the backbone; groups that land in `Uncategorized` are flagged | `taxonomy.map_lexicon_to_taxonomy` |
| Propose domains | Optional (`propose_domains: true`): the chat model suggests new domains for uncovered groups, which are added to the draft taxonomy | `propose.propose_domains` |
| Write | Schema-valid draft files for human review | `lexicon.save_lexicon`, `taxonomy.save_taxonomy` |

## Outputs

| File | Contents |
|---|---|
| `lexicon_draft.json`, `lexicon_draft.csv` | The grounded lexicon — edit either, then pass it to the dictionary workflow with `--lexicon` |
| `taxonomy_draft.json` | The backbone plus any proposed domains — pass with `--taxonomy` |
| `proposed_groups.csv` | The model's draft before grounding |
| `term_grounding.csv` | Every proposed term: keyword count, documents containing it, best passage similarity (semantic only), kept or not |
| `group_coverage.csv` | Per group: terms proposed vs kept, primary domain, and `needs_review` (dropped or uncovered) |
| `summary.md`, `manifest.json`, `resolved_config.yaml` | Run record, including the problem statement |

## Configuration

`RunConfig.formulation`: `problem_statement`, `max_groups`, `max_terms_per_group`,
`grounding` (`keyword` | `semantic` | `none`), `min_term_count`,
`min_terms_per_group`, `propose_domains`, `lexicon_name`. `chat` is required to
draft groups or propose domains; `embedding` is needed for semantic grounding or
`scoring.mapping_method: embedding`. Matching and mapping reuse the `scoring`
section, so grounding counts terms exactly the way the dictionary workflow will.

## Limits

- Grounding confirms that a term occurs in the seed corpus; it cannot tell
  whether the term is *missing* from the draft. Read the corpus's frequent terms
  (or run the narrative workflow) to find gaps.
- A small seed corpus will drop real but rare terms. Lower `min_term_count`,
  use `grounding: semantic`, or set `grounding: none` and review by hand.
- Model drafts vary between runs; the manifest records the model and settings,
  not a seed.
