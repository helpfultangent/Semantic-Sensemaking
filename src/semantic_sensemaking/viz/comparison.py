"""Base-layer vs projected attention bar chart."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from semantic_sensemaking.errors import optional_import
from semantic_sensemaking.viz.palette import NEUTRAL_GRAY, TACC_ORANGE

if TYPE_CHECKING:
    import plotly.graph_objects as go


def render_distribution_comparison(
    comparison: pd.DataFrame,
    *,
    title: str = "Domain attention — base layer vs. overlay projection",
    base_name: str = "Base (keyword groups)",
    overlay_name: str = "Overlay (projected)",
    hot: str = TACC_ORANGE,
) -> go.Figure:
    """Horizontal grouped bars from :func:`~semantic_sensemaking.taxonomy.compare_distributions`."""
    go = optional_import("plotly.graph_objects", "viz", "Rendering figures")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=comparison["domain"], x=comparison["base_pct"], orientation="h", name=base_name,
            marker=dict(color=NEUTRAL_GRAY), hovertemplate="%{y}<br>Base: %{x:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=comparison["domain"], x=comparison["overlay_pct"], orientation="h", name=overlay_name,
            marker=dict(color=hot), hovertemplate="%{y}<br>Overlay: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="% of attention",
        yaxis_title="",
        barmode="group",
        height=max(380, 28 * len(comparison) + 120),
        plot_bgcolor="white",
        margin=dict(l=240, r=40, t=70, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig
