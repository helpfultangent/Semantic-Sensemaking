"""Write a formulation to disk: draft lexicon and taxonomy, evidence tables, summary, manifest."""

from __future__ import annotations

import logging

from semantic_sensemaking.core.config import RunConfig
from semantic_sensemaking.core.context import RunContext
from semantic_sensemaking.lexicon.io import save_lexicon
from semantic_sensemaking.report import (
    file_sha256,
    markdown_table,
    render_summary,
    write_csv,
    write_manifest,
)
from semantic_sensemaking.taxonomy.io import save_taxonomy
from semantic_sensemaking.workflows.formulation.result import FormulationResult

log = logging.getLogger(__name__)


def write_outputs(result: FormulationResult, ctx: RunContext, config: RunConfig) -> None:
    for name, frame in result.tables().items():
        write_csv(frame, ctx.artifact(f"{name}.csv"))
    if len(result.lexicon):
        save_lexicon(result.lexicon, ctx.artifact("lexicon_draft.json"))
        save_lexicon(result.lexicon, ctx.artifact("lexicon_draft.csv"))
    else:
        log.warning("No group survived grounding; no lexicon draft written (see term_grounding.csv)")
    save_taxonomy(result.taxonomy, ctx.artifact("taxonomy_draft.json"))
    ctx.artifact("resolved_config.yaml").write_text(config.to_yaml(), encoding="utf-8")
    ctx.artifact("summary.md").write_text(_summary(result, ctx, config), encoding="utf-8")

    corpus = result.corpus
    write_manifest(
        ctx,
        config=config.snapshot(),
        inputs={
            "problem_statement": result.problem_statement,
            "corpus_root": str(corpus.root) if corpus else None,
            "documents": [
                {"doc_id": d.doc_id, "sha256": file_sha256(d.source_path) if d.source_path else None}
                for d in (corpus or ())
            ],
            "lexicon": (
                {"path": str(config.inputs.lexicon), "sha256": file_sha256(config.inputs.lexicon)}
                if config.inputs.lexicon
                else {"drafted_by": "chat model"}
            ),
            "taxonomy": (
                {"path": str(config.inputs.taxonomy), "sha256": file_sha256(config.inputs.taxonomy)}
                if config.inputs.taxonomy
                else {"builtin": result.taxonomy.source}
            ),
        },
        stats={
            "groups_proposed": len(result.proposed),
            "groups_kept": len(result.lexicon),
            "terms_proposed": sum(len(e.terms) for e in result.proposed),
            "terms_kept": sum(len(e.terms) for e in result.lexicon),
            "groups_uncovered": len(result.uncovered_groups()),
            "domains_added": len(result.added_domains),
        },
    )


def _summary(result: FormulationResult, ctx: RunContext, config: RunConfig) -> str:
    f = config.formulation
    coverage = result.coverage_table()
    review = coverage[coverage["needs_review"]][["group", "n_terms_kept", "primary_domain"]]
    added = "\n".join(
        f"- **{n.domain}**: {', '.join(n.subdisciplines) or '(no subdisciplines)'}"
        for n in result.added_domains
    ) or "_(none)_"
    outputs = "\n".join(f"- `{name}`" for name in ctx.relative_artifacts() + ["summary.md", "manifest.json"])
    corpus = result.corpus
    return render_summary(
        "Problem formulation — run summary",
        [
            ("Run", ctx.run_id),
            ("Problem statement", result.problem_statement or "_(checking an existing lexicon)_"),
            ("Seed corpus", f"{len(corpus)} document(s) from `{corpus.root}`" if corpus else "none"),
            ("Grounding", f"{f.grounding} (min_term_count={f.min_term_count})"),
            (
                "Groups",
                f"{len(result.proposed)} proposed → {len(result.lexicon)} kept "
                f"({len(result.dropped_groups)} dropped, {len(result.uncovered_groups())} uncovered)",
            ),
            ("Taxonomy", f"{result.taxonomy.source or result.taxonomy.name} ({len(result.taxonomy)} domains)"),
        ],
        [
            ("Draft lexicon", markdown_table(result.lexicon_table()) if len(result.lexicon) else "_(empty)_"),
            ("Needs review", markdown_table(review)),
            ("Proposed domains", added),
            (
                "Next step",
                "These files are drafts. Edit `lexicon_draft.json` (or `.csv`) and "
                "`taxonomy_draft.json`, then run:\n\n"
                "```bash\nsemantic-sensemaking dictionary --corpus <your corpus> \\\n"
                "  --lexicon lexicon_draft.json --taxonomy taxonomy_draft.json\n```",
            ),
            ("Outputs", outputs),
        ],
    )
