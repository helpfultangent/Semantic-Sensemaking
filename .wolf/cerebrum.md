---
description: learned preferences, project conventions, and Do-Not-Repeat rules
budget_tokens: 2000
---
# Cerebrum

> OpenWolf's learning memory. Updated automatically as the AI learns from interactions.
> Do not edit manually unless correcting an error.
> Last updated: 2026-11-18

## User Preferences

<!-- How the user likes things done. Code style, tools, patterns, communication. -->

## Key Learnings

- **Project:** semantic-sensemaking
- **Description:** This repository is the starting point for a Python library and executable examples for semantic sensemaking workflows. The original TACC cookbook deployment instructions are retained below while the l

- **Workflow contract:** each `workflows/<name>/` exposes DESCRIPTION, IMPLEMENTED, run(config); pure analysis fn (`analyze`/`formulate`) has no I/O; outputs.py writes tables + summary.md + manifest.json via `report`. Corpus loading shared in `workflows/_inputs.py`.
- **Config changes:** after editing `core/config.py`, regenerate schemas: `semantic-sensemaking schema src/semantic_sensemaking/data/schemas` (test_layering checks they're current).
- **Tapis apps:** `apps/<workflow>/app.json`; `tests/unit/test_apps.py` validates every SSM_* env var against RunConfig. Docker image has no PyTorch — TACC embeddings go through openai_compatible (vLLM/Ollama).
- **Offline testing:** `embedding.kind=hashing` + fake chat backends (objects with `complete_json`) keep all tests network-free.
- **CI:** test.yaml runs `ruff check src tests` and nbmake on notebooks/01_dictionary_overlay.ipynb (notebook missing as of 2026-11-18).

## Do-Not-Repeat

<!-- Mistakes made and corrected. Each entry prevents the same mistake recurring. -->
<!-- Format: [YYYY-MM-DD] Description of what went wrong and what to do instead. -->

## Decision Log

<!-- Significant technical decisions with rationale. Why X was chosen over Y. -->
