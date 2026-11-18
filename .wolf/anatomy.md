# anatomy.md

> Auto-maintained by OpenWolf. Last scanned: 2026-11-18T17:36:18.636Z
> Files: 106 tracked | Anatomy hits: 0 | Misses: 0

> Project structure index. Auto-maintained by OpenWolf hooks and daemon.
> Run `openwolf scan` to generate, or wait for the first Claude Code session.
> Status: Pending initial scan

## ./

- `.gitattributes` — Git attributes (~18 tok)
- `.gitignore` — Git ignore rules (~242 tok)
- `.pre-commit-config.yaml` — Install once with: pre-commit install (~100 tok)
- `app.json` (~819 tok)
- `CITATION.cff` — This CITATION.cff file was generated with cffinit. (~245 tok)
- `CLAUDE.md` — OpenWolf (~99 tok)
- `Dockerfile` — Docker container definition (~153 tok)
- `environment.yaml` (~40 tok)
- `package.json` — Node.js package manifest (~16 tok)
- `pyproject.toml` — Layered tools and TACC cookbooks for semantic sensemaking workflows. (~404 tok)
- `README.md` — Project documentation (~1801 tok)
- `run.sh` (~36 tok)

## .github/ISSUE_TEMPLATE/

- `bug_report.md` — 🐛 Bug Report (~154 tok)
- `config.yml` — Configuration: https://help.github.com/en/github/building-a-strong-community/configuring-issue-templates-for-your-repository (~45 tok)
- `feature_request.md` — 🚀 Feature Request (~128 tok)
- `question.md` — ❓ Question (~76 tok)

## .github/workflows/

- `docker.yaml` — CI: Docker image (~584 tok)
- `test.yaml` — CI: Test (~204 tok)

## apps/dictionary/

- `app.json` (~1926 tok)

## configs/

- `dictionary-corrected.yaml` — Dictionary workflow with the three prototype scoring issues fixed. (~182 tok)
- `dictionary-example.yaml` — Dictionary workflow job file. (~479 tok)
- `dictionary-semantic.yaml` — Dictionary workflow with embedding-based matching. (~291 tok)
- `lexicon-template.csv` (~91 tok)
- `lexicon-template.json` (~192 tok)

## docker/

- `Dockerfile` — Docker container definition (~350 tok)
- `environment.yaml` — Runtime environment for the container image. Python packages come from (~91 tok)
- `run.sh` — Tapis entrypoint shared by every Semantic Sensemaking app. (~383 tok)

## src/semantic_sensemaking/

- `__init__.py` — Semantic Sensemaking: layered tools for semantic workflows on TACC. (~413 tok)
- `_version.py` (~66 tok)
- `cli.py` — ``semantic-sensemaking`` command line: one entrypoint, one subcommand per workflow. (~1607 tok)
- `errors.py` — Exception hierarchy shared by every tool cluster. (~600 tok)

## src/semantic_sensemaking/backends/

- `__init__.py` — Embedding and chat backends behind shared protocols. (~317 tok)
- `anthropic_claude.py` — Claude chat backend (``pip install 'semantic-sensemaking[llm]'``). (~1248 tok)
- `base.py` — Backend protocols shared by every workflow. (~864 tok)
- `cache.py` — On-disk embedding cache, so reruns over the same corpus don't re-embed it. (~675 tok)
- `factory.py` — Build backends from :class:`~semantic_sensemaking.core.config.BackendConfig`. (~996 tok)
- `hashing.py` — Deterministic, dependency-free embeddings for tests and offline dry runs. (~392 tok)
- `huggingface_local.py` — Local sentence-transformers embeddings (``pip install 'semantic-sensemaking[embed]'``). (~380 tok)
- `openai_compatible.py` — Backend for any OpenAI-compatible HTTP API: vLLM or Ollama on TACC, or a hosted service. (~1920 tok)

## src/semantic_sensemaking/core/

- `__init__.py` — Shared data types, run configuration, and run context. (~219 tok)
- `config.py` — Run configuration shared by the CLI, notebooks, and Tapis apps. (~2532 tok)
- `context.py` — Per-run output directory and artifact bookkeeping. (~495 tok)
- `text.py` — Small text utilities shared across layers. (~353 tok)
- `types.py` — Core data types shared by every tool cluster and workflow. (~2694 tok)

## src/semantic_sensemaking/corpus/

- `__init__.py` — Turn files and directories into a :class:`~semantic_sensemaking.core.types.Corpus`. (~153 tok)
- `json_text.py` — Extract prose from transcript-shaped JSON. (~306 tok)
- `loader.py` — Load a file or directory into a :class:`~semantic_sensemaking.core.types.Corpus`. (~1068 tok)
- `readers.py` — File readers keyed by suffix. Register more with :func:`register_reader`. (~705 tok)
- `select.py` — Resolve a user-supplied document spec to one document. (~513 tok)

## src/semantic_sensemaking/data/

- `__init__.py` — Package data: built-in lexicons, taxonomies, JSON schemas, and a sample corpus. (~182 tok)
- `lexicon_subside_v1.json` (~702 tok)
- `taxonomy_subside_v1.json` (~494 tok)

## src/semantic_sensemaking/data/sample_corpus/

- `aquifer_management_panel.json` (~358 tok)
- `coastal_subsidence_briefing.md` — Coastal subsidence briefing (synthetic example) (~261 tok)
- `insar_methods_lecture.json` (~349 tok)
- `oilfield_compaction_notes.txt` (~253 tok)
- `urban_flood_workshop.txt` (~247 tok)

## src/semantic_sensemaking/lexicon/

- `__init__.py` — User-defined dictionaries: load, match terms, and score corpora. (~200 tok)
- `io.py` — Read and write user-defined dictionaries (lexicons). (~1926 tok)
- `matching.py` — Count lexicon terms in text. (~1191 tok)
- `scoring.py` — Score a corpus against a lexicon: the document × group count table. (~1213 tok)
- `semantic.py` — Embedding-based lexicon matching. (~1119 tok)

## src/semantic_sensemaking/report/

- `__init__.py` — Writers shared by every workflow: tables, Markdown summaries, run manifests. (~702 tok)

## src/semantic_sensemaking/taxonomy/

- `__init__.py` — Science backbones: load, map concepts onto them, and project weights. (~234 tok)
- `io.py` — Read and write taxonomies (science backbones). (~1912 tok)
- `projection.py` — Project label weights onto a taxonomy and compare distributions. (~1145 tok)

## src/semantic_sensemaking/viz/

- `__init__.py` — Figures for semantic workflows (``pip install 'semantic-sensemaking[viz]'``). (~437 tok)
- `backbone_map.py` — Radial "soft petal" backbone overlay figure. (~2879 tok)
- `comparison.py` — Base-layer vs projected attention bar chart. (~478 tok)
- `geometry.py` — Radial layout and polygon helpers for backbone figures (no plotting dependency). (~1769 tok)
- `heatmap.py` — Document × group heatmap. (~376 tok)
- `overlay_data.py` — The neutral data contract consumed by the backbone overlay renderer. (~375 tok)
- `palette.py` — Colors and domain ordering for backbone figures. (~859 tok)

## src/semantic_sensemaking/workflows/

- `__init__.py` — End-to-end workflows built from the shared tool clusters. (~217 tok)
- `_inputs.py` — Input loading shared by workflows. (~352 tok)

## src/semantic_sensemaking/workflows/dictionary/

- `__init__.py` — User-defined dictionary-driven semantic analysis. (~210 tok)
- `outputs.py` — Write a dictionary analysis to disk: tables, figures, summary, manifest. (~1598 tok)
- `pipeline.py` — Dictionary workflow orchestration: load inputs, analyze, write outputs. (~1076 tok)
- `result.py` — The in-memory result of a dictionary analysis. (~1246 tok)

## src/semantic_sensemaking/workflows/formulation/

- `__init__.py` — Semantic problem formulation — planned. See ``README.md`` in this package. (~153 tok)
- `README.md` — Project documentation (~431 tok)

## src/semantic_sensemaking/workflows/narrative/

- `__init__.py` — Narrative analysis: find the frames a corpus uses and place them on a science backbone. (~348 tok)
- `cluster.py` — Group passage embeddings into candidate frames with spherical k-means. (~1162 tok)
- `keywords.py` — Distinctive terms per cluster (class-based TF-IDF). (~1128 tok)
- `labeling.py` — Turn clusters into named frames, with a chat model or from keywords alone. (~1448 tok)
- `outputs.py` — Write a narrative analysis to disk: tables, frame lexicon, figures, summary, manifest. (~1644 tok)
- `pipeline.py` — Narrative workflow orchestration: segment, embed, cluster, label, map, project. (~1546 tok)
- `README.md` — Project documentation (~917 tok)
- `result.py` — The in-memory result of a narrative analysis. (~2394 tok)

## tests/

- `conftest.py` — sample_corpus, subside_lexicon, subside_taxonomy, write_pdf (~914 tok)

## tests/integration/

- `test_cli_workflow.py` — End-to-end runs through the same entrypoint the Tapis app uses. (~1061 tok)

## tests/parity/

- `__init__.py` (~0 tok)
- `make_golden.py` — Regenerate ``golden_sample_corpus.json`` from the prototype oracle. (~370 tok)
- `prototype_reference.py` — Test oracle: the SUBSIDE Backbone Overlay prototype's analysis, ported verbatim. (~2480 tok)
- `test_bug_fixes.py` — Each prototype scoring bug, shown under legacy settings and fixed under the opt-in setting. (~1219 tok)
- `test_prototype_parity.py` — The library with legacy settings reproduces the prototype notebook's numbers. (~1000 tok)

## tests/unit/

- `test_backends.py` — Recorder: test_hashing_embedding_is_deterministic_normalized_and_word_sensitive, embeddings_response (~1965 tok)
- `test_config.py` — write_yaml, test_defaults_are_valid, test_precedence_yaml_then_env_then_overrides, test_env_lists_an (~706 tok)
- `test_corpus.py` — test_sample_corpus_loads_every_file, test_edge_corpus_reads_all_formats_and_reports_skips, test_shar (~1242 tok)
- `test_layering.py` — Architectural rules for the tool-cluster hierarchy. (~668 tok)
- `test_lexicon_and_taxonomy_io.py` — test_builtin_lexicon_and_taxonomy, test_flat_lexicon_json_with_comment_keys, test_csv_lexicon_preser (~1271 tok)
- `test_matching.py` — test_auto_uses_word_boundaries_for_single_words, test_auto_substring_matches_inside_words_for_multiw (~542 tok)
- `test_narrative.py` — Narrative workflow: segmentation, clustering, keywords, labeling, projection. (~1540 tok)
- `test_semantic.py` — test_semantic_scores_count_matching_passages, test_threshold_above_one_matches_nothing, test_explain (~710 tok)
- `test_taxonomy_projection.py` — test_no_hits_maps_to_uncategorized, test_top_n_limits_legacy_assignments, test_projection_splits_wei (~674 tok)
- `test_viz.py` — test_hex_to_rgba, test_heat_color_endpoints, test_geometry_helpers, test_domain_order_puts_known_dom (~857 tok)
