from __future__ import annotations

import subprocess
import sys

import pytest

from semantic_sensemaking.viz.geometry import convex_hull, fan, petal_outline
from semantic_sensemaking.viz.palette import heat_color, hex_to_rgba, order_domains
from semantic_sensemaking.workflows.dictionary import analyze


@pytest.mark.parametrize(
    ("color", "expected"),
    [("#abc", "rgba(170,187,204,0.5)"), ("#AABBCC", "rgba(170,187,204,0.5)"), ("rgb(1, 2, 3)", "rgba(1,2,3,0.5)")],
)
def test_hex_to_rgba(color, expected):
    assert hex_to_rgba(color, 0.5) == expected


def test_heat_color_endpoints():
    assert heat_color(0, 10) == "#F5F5F5"
    assert heat_color(10, 10) == heat_color(99, 10) == "#BF5700"
    assert heat_color(5, 0) == "#F5F5F5"


def test_geometry_helpers():
    square = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
    assert sorted(convex_hull(square)) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert len(fan(0.0, 5, 1.0, 2.0)) == 5
    assert fan(0.0, 1, 1.0, 2.0) == [(2.0, 0.0)]
    assert len(petal_outline([(0, 0), (1, 0)])) == 4
    assert petal_outline([(0, 0), (0, 0)]) == []


def test_domain_order_puts_known_domains_first():
    assert order_domains(["Zoology", "Earth Sciences", "Social Sciences"]) == [
        "Social Sciences", "Earth Sciences", "Zoology",
    ]


def test_importing_viz_does_not_import_plotly():
    code = "import sys, semantic_sensemaking.viz, semantic_sensemaking.workflows.dictionary; print('plotly' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"


@pytest.fixture(scope="module")
def result(sample_corpus, subside_lexicon, subside_taxonomy):
    return analyze(sample_corpus, subside_lexicon, subside_taxonomy)


def test_corpus_overlay_figure(result):
    from semantic_sensemaking.viz import render_backbone_overlay

    fig = render_backbone_overlay(result.to_overlay_data())
    names = {trace.name for trace in fig.data}
    assert "Documents (5)" in names and "Domain" in names
    assert any(name and name.startswith("Group → ") for name in names)


def test_single_document_figure_labels_include_counts(result):
    from semantic_sensemaking.viz import SINGLE_DOCUMENT_STYLE, render_backbone_overlay

    fig = render_backbone_overlay(result.to_overlay_data("oilfield"), SINGLE_DOCUMENT_STYLE)
    labels = [t for trace in fig.data if trace.mode == "markers+text" and trace.name != "Domain" for t in trace.text]
    assert labels and all(label.endswith(")") for label in labels)
    assert not any((trace.name or "").startswith("Documents") for trace in fig.data)
    assert fig.layout.height == 780


def test_comparison_and_heatmap_figures(result):
    from semantic_sensemaking.viz import render_distribution_comparison, render_doc_group_heatmap

    assert len(render_distribution_comparison(result.comparison).data) == 2
    heatmap = render_doc_group_heatmap(result.scores.pivot())
    assert heatmap.data[0].z.shape == (5, 18)
