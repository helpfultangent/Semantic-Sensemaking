"""Document × group heatmap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from semantic_sensemaking.errors import optional_import
from semantic_sensemaking.viz.palette import TACC_ORANGE

if TYPE_CHECKING:
    import plotly.graph_objects as go


def render_doc_group_heatmap(
    pivot: pd.DataFrame,
    *,
    title: str = "Per-document × per-keyword-group projection (normalized per 1k words)",
    colorbar_title: str = "matches<br>per 1k words",
    hot: str = TACC_ORANGE,
) -> go.Figure:
    """Heatmap of a ``LexiconScores.pivot()`` matrix."""
    go = optional_import("plotly.graph_objects", "viz", "Rendering figures")
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[[0, "#f7f7f7"], [1, hot]],
            colorbar=dict(title=colorbar_title),
            hovertemplate="Doc: %{y}<br>Group: %{x}<br>%{z:.2f}/1k words<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Keyword group",
        yaxis_title="Document",
        height=max(360, 22 * len(pivot) + 200),
        plot_bgcolor="white",
        margin=dict(l=220, r=40, t=70, b=120),
        xaxis=dict(tickangle=-35),
    )
    return fig
