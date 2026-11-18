"""Write a narrative analysis to disk: tables, frame lexicon, figures, summary, manifest."""

from __future__ import annotations

import logging

from semantic_sensemaking.core.config import RunConfig
from semantic_sensemaking.core.context import RunContext
from semantic_sensemaking.core.text import safe_filename
from semantic_sensemaking.errors import MissingExtraError
from semantic_sensemaking.lexicon.io import save_lexicon
from semantic_sensemaking.report import (
    file_sha256,
    markdown_table,
    render_summary,
    write_csv,
    write_manifest,
)
from semantic_sensemaking.workflows.narrative.result import NarrativeResult

log = logging.getLogger(__name__)


def write_outputs(result: NarrativeResult, ctx: RunContext, config: RunConfig) -> None:
    for name, frame in result.tables().items():
        write_csv(frame, ctx.artifact(f"{name}.csv"))
    save_lexicon(result.lexicon, ctx.artifact("frames_lexicon.json"))
    ctx.artifact("resolved_config.yaml").write_text(config.to_yaml(), encoding="utf-8")

    if config.viz.enabled:
        try:
            _write_figures(result, ctx, config)
        except MissingExtraError as exc:
            log.warning("Skipping figures: %s", exc)

    ctx.artifact("summary.md").write_text(_summary(result, ctx, config), encoding="utf-8")
    write_manifest(
        ctx,
        config=config.snapshot(),
        inputs={
            "corpus_root": str(result.corpus.root),
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "sha256": file_sha256(d.source_path) if d.source_path else None,
                    "n_words": d.n_words,
                }
                for d in result.corpus
            ],
            "taxonomy": (
                {"path": str(config.inputs.taxonomy), "sha256": file_sha256(config.inputs.taxonomy)}
                if config.inputs.taxonomy
                else {"builtin": result.taxonomy.source}
            ),
        },
        stats={
            "documents": len(result.corpus),
            "skipped_files": len(result.corpus.skipped),
            "passages": len(result.passages),
            "frames": len(result.frames),
            "frames_labeled_by_chat": sum(f.labeled_by == "chat" for f in result.frames),
            "taxonomy_domains": len(result.taxonomy),
        },
    )


def _write_figures(result: NarrativeResult, ctx: RunContext, config: RunConfig) -> None:
    from semantic_sensemaking.viz import (
        SINGLE_DOCUMENT_STYLE,
        render_backbone_overlay,
        render_distribution_comparison,
        render_doc_group_heatmap,
        write_figure,
    )

    plotlyjs = config.viz.include_plotlyjs
    write_figure(
        render_backbone_overlay(result.to_overlay_data()),
        ctx.artifact("narrative_backbone.html"),
        include_plotlyjs=plotlyjs,
    )
    write_figure(
        render_distribution_comparison(result.comparison),
        ctx.artifact("narrative_vs_base.html"),
        include_plotlyjs=plotlyjs,
    )
    pivot = result.pivot()
    if not pivot.empty:
        write_figure(
            render_doc_group_heatmap(
                pivot,
                title="Share of each document's passages in each frame",
                colorbar_title="% of<br>passages",
            ),
            ctx.artifact("narrative_per_doc_heatmap.html"),
            include_plotlyjs=plotlyjs,
        )

    if not config.viz.per_document_figures:
        return
    for written, doc in enumerate(result.corpus):
        if written >= config.viz.max_per_document_figures:
            log.warning("Stopped after %d per-document figures (viz.max_per_document_figures)", written)
            break
        write_figure(
            render_backbone_overlay(result.to_overlay_data(doc.doc_id), SINGLE_DOCUMENT_STYLE),
            ctx.artifact(f"per_document/narrative_single_{safe_filename(doc.doc_id)}.html"),
            include_plotlyjs=plotlyjs,
        )


def _summary(result: NarrativeResult, ctx: RunContext, config: RunConfig) -> str:
    n = config.narrative
    frames = result.frames_table()[["frame_id", "frame", "n_passages", "n_documents", "primary_domain", "keywords"]]
    skipped = "\n".join(f"- `{s.path}`: {s.reason}" for s in result.corpus.skipped[:20]) or "_(none)_"
    outputs = "\n".join(f"- `{name}`" for name in ctx.relative_artifacts() + ["summary.md", "manifest.json"])
    labeled = "chat model" if any(f.labeled_by == "chat" for f in result.frames) else "keywords only"
    return render_summary(
        "Narrative frames — run summary",
        [
            ("Run", ctx.run_id),
            ("Corpus", f"{len(result.corpus)} document(s) from `{result.corpus.root}`"),
            ("Passages", f"{len(result.passages)} of ~{n.segment_words} words"),
            ("Frames", f"{len(result.frames)} (max {n.max_frames}, min {n.min_frame_passages} passages each)"),
            ("Frame labels", labeled),
            ("Taxonomy", f"{result.taxonomy.source or result.taxonomy.name} ({len(result.taxonomy)} domains)"),
            ("Mapping", f"method={result.scoring.mapping_method}, unmapped_groups={result.scoring.unmapped_groups}"),
        ],
        [
            ("Frames", markdown_table(frames)),
            ("Top 5 domains by projected weight", markdown_table(result.projection.domain_totals().head(5))),
            (
                "Next step",
                "`frames_lexicon.json` holds the frames as a lexicon. Review and edit it, then "
                "run it through the dictionary workflow (`--lexicon frames_lexicon.json`) to "
                "count the frames by keyword across this or another corpus.",
            ),
            ("Skipped files", skipped),
            ("Outputs", outputs),
        ],
    )
