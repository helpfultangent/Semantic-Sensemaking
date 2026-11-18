# Narrative analysis workflow (planned)

**Goal:** find the stories a corpus tells — recurring frames, actors, claims, and
how they shift across documents — and place those frames on the same science
backbone the dictionary workflow uses, so dictionary and narrative results can be
compared on one reference frame.

## Intended pipeline

| Step | What it does | Shared layer it reuses |
|---|---|---|
| Ingest | Same transcripts, PDFs, and notes | `corpus.load_corpus` |
| Segment | Split documents into passages of `narrative.segment_words` | `core.text.chunk_words` |
| Embed | Vectorize passages | `backends.get_embedding_backend` (+ `CachedEmbeddingBackend`) |
| Cluster | Group passages into candidate frames | new: `narrative/cluster.py` |
| Label | Ask a chat model to name each frame and list its signal terms, as structured output | `ChatBackend.complete_json` |
| Map | Frames become lexicon-like `{frame: terms}` and are mapped onto the backbone | `taxonomy.map_lexicon_to_taxonomy` (term or embedding method) |
| Project | Frame weights (passages per frame) projected onto domains | `taxonomy.project_weights` — unchanged |
| Render | Backbone overlay with frames in place of keyword groups | `viz.BackboneOverlayData` + `render_backbone_overlay` — unchanged |
| Report | Tables, summary, manifest | `report` |

Only clustering and frame labeling are new code; everything else is already in
the library. Because frames can be exported as a `Lexicon`, a labeled frame set
can also be reviewed by a person and re-run through the dictionary workflow.

## Configuration

`RunConfig.narrative` (`segment_words`, `max_frames`) is reserved now so job files
written against this design stay valid. `embedding` and `chat` configure the
backends.
