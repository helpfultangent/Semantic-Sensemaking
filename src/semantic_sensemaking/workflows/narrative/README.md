# Narrative analysis workflow

**Goal:** find the stories a corpus tells — recurring frames, actors, and claims —
and place those frames on the same science backbone the dictionary workflow uses,
so dictionary and narrative results can be compared on one reference frame.

```bash
semantic-sensemaking narrative --config configs/narrative-example.yaml --corpus path/to/docs
# offline smoke test, no models:
semantic-sensemaking narrative --sample-corpus --set embedding.kind=hashing --set narrative.segment_words=40
```

```python
from semantic_sensemaking import load_corpus, load_taxonomy
from semantic_sensemaking.backends import get_embedding_backend
from semantic_sensemaking.workflows.narrative import analyze

result = analyze(corpus, load_taxonomy(), embedding_backend, narrative_config, chat_backend=chat)
result.frames_table()
```

## Pipeline

| Step | What it does | Code |
|---|---|---|
| Ingest | Same transcripts, PDFs, and notes as the dictionary workflow | `corpus.load_corpus` |
| Segment | Split documents into passages of `narrative.segment_words` | `pipeline.segment_corpus` → `core.text.chunk_words` |
| Embed | Vectorize passages (required) | `backends.get_embedding_backend` (+ cache) |
| Cluster | Spherical k-means, at most `max_frames` clusters of at least `min_frame_passages`; deterministic per `seed` | `cluster.py` (NumPy only) |
| Keywords | Distinctive unigrams/bigrams per cluster (class-based TF-IDF) | `keywords.py` |
| Label | Chat model names each frame from its most central passages (structured output); without a chat backend, or if a reply fails, the frame is named from its keywords | `labeling.py` → `ChatBackend.complete_json` |
| Map | Frames become a `Lexicon` (terms = model terms + keywords) and are mapped onto the backbone with `scoring.mapping_method` | `taxonomy.map_lexicon_to_taxonomy` |
| Project | Frame weights (passages per frame) projected onto domains | `taxonomy.project_weights` |
| Render | Backbone overlay with frames in place of keyword groups | `viz.render_backbone_overlay` |
| Report | Tables, frame lexicon, summary, manifest | `outputs.py` → `report` |

## Outputs

| File | Contents |
|---|---|
| `frames.csv` | One row per frame: name, description, size, documents, primary domain, terms, keywords, central passage |
| `passage_frames.csv` | Every passage, its frame, and its similarity to the frame centroid |
| `doc_frame_counts.csv` | Passages per document × frame, and each frame's share of the document |
| `frame_mapping.csv`, `frame_domain_totals.csv`, `frame_subdiscipline_totals.csv`, `comparison_base_vs_frames.csv` | Same as the dictionary workflow's tables, for frames |
| `frames_lexicon.json` | Frames as a lexicon — edit it, then run it through the dictionary workflow with `--lexicon` |
| `narrative_backbone.html`, `narrative_vs_base.html`, `narrative_per_doc_heatmap.html` | Figures (needs the `viz` extra) |
| `summary.md`, `manifest.json`, `resolved_config.yaml` | Run record |

## Configuration

`RunConfig.narrative`: `segment_words`, `segment_overlap_words`, `max_frames`,
`min_frame_passages`, `terms_per_frame`, `representative_passages`, `seed`.
`embedding` (required) and `chat` (optional) configure the backends. Mapping
reuses `scoring.mapping_method`, `scoring.top_n_assignments` and
`scoring.unmapped_groups`.

## Limits

- The number of frames is an upper bound, not chosen automatically; try a few
  values of `max_frames` and compare `frames.csv`.
- A passage belongs to exactly one frame. Overlapping frames within a passage
  are not modeled.
- With the `hashing` backend, frames reflect shared words, not meaning — use it
  only to check the pipeline runs.
