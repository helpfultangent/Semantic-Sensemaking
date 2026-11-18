"""Figures for semantic workflows (``pip install 'semantic-sensemaking[viz]'``).

Importing this package does not import plotly; the ``render_*`` functions do.
"""

from pathlib import Path
from typing import Any, Literal

from semantic_sensemaking.viz.backbone_map import (
    SINGLE_DOCUMENT_STYLE,
    OverlayStyle,
    render_backbone_overlay,
)
from semantic_sensemaking.viz.comparison import render_distribution_comparison
from semantic_sensemaking.viz.geometry import Radii, radial_layout
from semantic_sensemaking.viz.heatmap import render_doc_group_heatmap
from semantic_sensemaking.viz.overlay_data import BackboneOverlayData
from semantic_sensemaking.viz.palette import KLAVANS_BOYACK_ORDER, TACC_ORANGE, UCSD_HEX


def write_figure(
    fig: Any, path: str | Path, *, include_plotlyjs: Literal["cdn", "inline"] = "cdn"
) -> Path:
    """Write a plotly figure to standalone HTML.

    ``"cdn"`` keeps files small but needs internet to view; use ``"inline"`` for
    figures that will be opened on an offline machine.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn" if include_plotlyjs == "cdn" else True)
    return path


__all__ = [
    "KLAVANS_BOYACK_ORDER",
    "SINGLE_DOCUMENT_STYLE",
    "TACC_ORANGE",
    "UCSD_HEX",
    "BackboneOverlayData",
    "OverlayStyle",
    "Radii",
    "radial_layout",
    "render_backbone_overlay",
    "render_distribution_comparison",
    "render_doc_group_heatmap",
    "write_figure",
]
