"""End-to-end runs through the same entrypoint the Tapis app uses."""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd

from semantic_sensemaking import __version__, sample_corpus_path
from semantic_sensemaking.cli import main

TABLES = {
    "overlay_doc_group_counts.csv",
    "overlay_group_totals.csv",
    "overlay_domain_totals.csv",
    "overlay_subdiscipline_totals.csv",
    "comparison_base_vs_overlay.csv",
    "group_mapping.csv",
    "corpus_documents.csv",
    "skipped_files.csv",
}
FIGURES = {"overlay_backbone.html", "overlay_vs_base.html", "overlay_per_doc_heatmap.html"}


def test_dictionary_run_writes_every_artifact(tmp_path):
    code = main(
        [
            "dictionary", "--sample-corpus", "--output-dir", str(tmp_path), "--run-id", "t1",
            "--set", "viz.per_document_figures=true",
        ]
    )
    assert code == 0
    run = tmp_path / "t1"
    assert TABLES | FIGURES | {"resolved_config.yaml", "summary.md", "manifest.json"} <= {
        p.name for p in run.iterdir()
    }
    assert len(list((run / "per_document").glob("overlay_single_*.html"))) == 5
    assert len(pd.read_csv(run / "overlay_doc_group_counts.csv")) == 5 * 18

    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["stats"]["documents"] == 5
    assert manifest["inputs"]["lexicon"] == {"builtin": "built-in 'subside_v1'"}
    assert all(d["sha256"] for d in manifest["inputs"]["documents"])
    assert set(manifest["artifacts"]) >= TABLES | FIGURES

    summary = (run / "summary.md").read_text()
    assert "## Top 5 groups" in summary and "match_mode=auto" in summary


def test_user_dictionary_job_file_and_no_figures(tmp_path):
    lexicon = tmp_path / "my_terms.csv"
    lexicon.write_text("group,term\nWater,aquifer\nWater,groundwater\nPeople,stakeholder\n")
    job = tmp_path / "job.yaml"
    job.write_text(
        f"inputs:\n  corpus: {sample_corpus_path()}\n  lexicon: {lexicon}\n"
        "scoring:\n  match_mode: longest_match\n  unmapped_groups: keep\nviz:\n  enabled: false\n"
    )
    assert main(["dictionary", "--config", str(job), "--output-dir", str(tmp_path / "out"), "--run-id", "r"]) == 0
    run = tmp_path / "out" / "r"
    totals = pd.read_csv(run / "overlay_group_totals.csv")
    assert set(totals["group"]) == {"Water", "People"}
    assert not list(run.glob("*.html"))
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["inputs"]["lexicon"]["path"] == str(lexicon)
    assert manifest["config"]["scoring"]["match_mode"] == "longest_match"


def test_config_show_prints_resolved_yaml(capsys, monkeypatch):
    monkeypatch.setenv("SSM_CORPUS__MAX_FILES", "7")
    assert main(["config", "show", "--sample-corpus", "--set", "scoring.match_mode=longest_match"]) == 0
    out = capsys.readouterr().out
    assert "match_mode: longest_match" in out and "max_files: 7" in out


def test_exit_codes(tmp_path, capsys):
    assert main(["dictionary", "--output-dir", str(tmp_path)]) == 2
    assert "No corpus" in capsys.readouterr().err
    assert main(["dictionary", "--corpus", str(tmp_path / "missing"), "--output-dir", str(tmp_path)]) == 1
    assert main(["dictionary", "--sample-corpus", "--set", "scoring.matchmode=auto"]) == 2
    capsys.readouterr()
    assert main(["narrative", "--sample-corpus"]) == 2
    assert "embedding backend" in capsys.readouterr().err
    assert main(["formulate", "--sample-corpus"]) == 2
    assert "problem_statement" in capsys.readouterr().err


def test_module_entrypoint_version():
    out = subprocess.run(
        [sys.executable, "-m", "semantic_sensemaking.cli", "--version"], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == f"semantic-sensemaking {__version__}"


def test_narrative_run_writes_every_artifact_and_feeds_dictionary(tmp_path):
    code = main(
        [
            "narrative", "--sample-corpus", "--output-dir", str(tmp_path), "--run-id", "n1",
            "--set", "embedding.kind=hashing", "--set", "narrative.segment_words=40",
            "--set", "viz.per_document_figures=true",
        ]
    )
    assert code == 0
    run = tmp_path / "n1"
    expected = {
        "frames.csv", "passage_frames.csv", "doc_frame_counts.csv", "frame_mapping.csv",
        "frame_domain_totals.csv", "frame_subdiscipline_totals.csv",
        "comparison_base_vs_frames.csv", "corpus_documents.csv", "skipped_files.csv",
        "frames_lexicon.json", "narrative_backbone.html", "narrative_vs_base.html",
        "narrative_per_doc_heatmap.html", "resolved_config.yaml", "summary.md", "manifest.json",
    }
    assert expected <= {p.name for p in run.iterdir()}
    assert len(list((run / "per_document").glob("narrative_single_*.html"))) == 5
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["workflow"] == "narrative"
    assert manifest["stats"]["frames"] == len(pd.read_csv(run / "frames.csv"))

    assert main(
        [
            "dictionary", "--sample-corpus", "--lexicon", str(run / "frames_lexicon.json"),
            "--output-dir", str(tmp_path), "--run-id", "d1", "--set", "viz.enabled=false",
        ]
    ) == 0


def test_formulation_checks_an_existing_lexicon_offline(tmp_path):
    """No chat model: ground the built-in lexicon against the sample corpus and write drafts."""
    from semantic_sensemaking.data import builtin_path

    code = main(
        [
            "formulation", "--sample-corpus", "--lexicon", str(builtin_path("lexicon_subside_v1.json")),
            "--output-dir", str(tmp_path), "--run-id", "f1",
        ]
    )
    assert code == 0
    run = tmp_path / "f1"
    assert {
        "proposed_groups.csv", "group_coverage.csv", "term_grounding.csv", "lexicon_draft.json",
        "lexicon_draft.csv", "taxonomy_draft.json", "summary.md", "manifest.json",
    } <= {p.name for p in run.iterdir()}
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["workflow"] == "formulation"
    assert 0 < manifest["stats"]["terms_kept"] <= manifest["stats"]["terms_proposed"]
    assert main(
        [
            "dictionary", "--sample-corpus", "--lexicon", str(run / "lexicon_draft.json"),
            "--taxonomy", str(run / "taxonomy_draft.json"), "--output-dir", str(tmp_path),
            "--run-id", "d2", "--set", "viz.enabled=false",
        ]
    ) == 0
