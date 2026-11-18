"""Write a dictionary analysis to disk: tables, figures, summary, manifest."""

from __future__ import annotations

import logging

from semantic_sensemaking.core.config import RunConfig
from semantic_sensemaking.core.context import RunContext
from semantic_sensemaking.core.text import safe_filename
from semantic_sensemaking.errors import MissingExtraError
from semantic_sensemaking.report import (
    file_sha256,
    markdown_table,
    render_summary,
    write_csv,
    write_manifest,
)
from semantic_sensemaking.workflows.dictionary.result import DictionaryResult

log = logging.getLogger(__name__)


def write_outputs(result: DictionaryResult, ctx: RunContext, config: RunConfig) -> None:
    for name, frame in result.frames().items():
        write_csv(frame, ctx.artifact(f"{name}.csv"))
    ctx.artifact("resolved_config.yaml").write_text(config.to_yaml(), encoding="utf-8")

    if config.viz.enabled:
        try:
            _write_figures(result, ctx, config)
        except MissingExtraError as exc:
            log.warning("Skipping figures: %s", exc)

    ctx.artifact("summary.md").write_text(_summary(result, ctx), encoding="utf-8")
    write_manifest(
        ctx,
        config=config.snapshot(),
        inputs=_inputs(result, config),
        stats={
            "documents": len(result.corpus),
            "skipped_files": len(result.corpus.skipped),
            "lexicon_groups": len(result.lexicon),
            "taxonomy_domains": len(result.taxonomy),
            "total_matches": result.scores.total_matches(),
        },
    )


def _write_figures(result: DictionaryResult, ctx: RunContext, config: RunConfig) -> None:
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
        ctx.artifact("overlay_backbone.html"),
        include_plotlyjs=plotlyjs,
    )
    write_figure(
        render_distribution_comparison(result.comparison),
        ctx.artifact("overlay_vs_base.html"),
        include_plotlyjs=plotlyjs,
    )
    pivot = result.scores.pivot()
    if not pivot.empty:
        write_figure(
            render_doc_group_heatmap(pivot),
            ctx.artifact("overlay_per_doc_heatmap.html"),
            include_plotlyjs=plotlyjs,
        )

    if not config.viz.per_document_figures:
        return
    written = 0
    for doc in result.corpus:
        if written >= config.viz.max_per_document_figures:
            log.warning(
                "Stopped after %d per-document figures (viz.max_per_document_figures)", written
            )
            break
        if result.scores.for_document(doc.doc_id).total_matches() == 0:
            log.info("No matches in %s; skipping its figure", doc.label)
            continue
        write_figure(
            render_backbone_overlay(result.to_overlay_data(doc.doc_id), SINGLE_DOCUMENT_STYLE),
            ctx.artifact(f"per_document/overlay_single_{safe_filename(doc.doc_id)}.html"),
            include_plotlyjs=plotlyjs,
        )
        written += 1


def _summary(result: DictionaryResult, ctx: RunContext) -> str:
    scoring = result.scoring
    gaps = result.comparison.reindex(
        result.comparison["delta_pct"].abs().sort_values(ascending=False).index
    ).head(5)[["domain", "base_pct", "overlay_pct", "delta_pct"]]
    skipped = "\n".join(f"- `{s.path}`: {s.reason}" for s in result.corpus.skipped[:20]) or "_(none)_"
    outputs = "\n".join(f"- `{name}`" for name in ctx.relative_artifacts() + ["summary.md", "manifest.json"])
    return render_summary(
        "Dictionary overlay — run summary",
        [
            ("Run", ctx.run_id),
            ("Lexicon", f"{result.lexicon.name} ({len(result.lexicon)} groups; source: {result.lexicon.source})"),
            ("Taxonomy", f"{result.taxonomy.source or result.taxonomy.name} ({len(result.taxonomy)} domains)"),
            ("Corpus", f"{len(result.corpus)} document(s) from `{result.corpus.root}`"),
            ("Skipped files", len(result.corpus.skipped)),
            (
                "Scoring",
                f"method={scoring.method}, match_mode={scoring.match_mode.value}, "
                f"mapping_method={scoring.mapping_method}, unmapped_groups={scoring.unmapped_groups}",
            ),
        ],
        [
            ("Top 5 groups", markdown_table(result.scores.group_totals().head(5))),
            ("Top 5 domains by projected weight", markdown_table(result.projection.domain_totals().head(5))),
            ("Largest base ↔ overlay attention gaps", markdown_table(gaps)),
            ("Skipped files", skipped),
            ("Outputs", outputs),
        ],
    )


def _inputs(result: DictionaryResult, config: RunConfig) -> dict[str, object]:
    def described(path: object, fallback: str | None) -> object:
        if path is None:
            return {"builtin": fallback}
        return {"path": str(path), "sha256": file_sha256(path)}

    return {
        "corpus_root": str(result.corpus.root),
        "documents": [
            {
                "doc_id": d.doc_id,
                "sha256": file_sha256(d.source_path) if d.source_path else None,
                "n_words": d.n_words,
            }
            for d in result.corpus
        ],
        "lexicon": described(config.inputs.lexicon, result.lexicon.source),
        "taxonomy": described(config.inputs.taxonomy, result.taxonomy.source),
    }
