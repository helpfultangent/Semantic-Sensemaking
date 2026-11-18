"""Radial "soft petal" backbone overlay figure.

One renderer replaces the prototype's two near-duplicates (the corpus-wide
figure in §5 and the single-document figure in §5.5). Differences between them
are now options: the outer document ring appears when ``outer_nodes`` is given,
weights appear in group labels with ``show_weights_in_labels``, and figure size
comes from :class:`OverlayStyle`. Opacity and heat scale against the true
maximum weight in the figure (the prototype floored the corpus figure's maximum
at 1.0, which washed out projections whose weights were all below 1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from semantic_sensemaking.errors import optional_import
from semantic_sensemaking.viz.geometry import (
    Radii,
    label_position,
    outer_key,
    petal_outline,
    radial_layout,
    subdiscipline_key,
)
from semantic_sensemaking.viz.overlay_data import BackboneOverlayData
from semantic_sensemaking.viz.palette import (
    HEAT_BASE,
    KLAVANS_BOYACK_ORDER,
    NEUTRAL_GRAY,
    TACC_ORANGE,
    domain_colors,
    heat_color,
    hex_to_rgba,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go


@dataclass(frozen=True)
class OverlayStyle:
    radii: Radii = Radii()
    arc_subdisciplines_deg: float = 24.0
    arc_groups_deg: float = 20.0
    hull_margin: float = 0.22
    hot: str = TACC_ORANGE
    heat_base: str = HEAT_BASE
    height: int = 820
    axis_range: float = 4.5
    domain_order: Sequence[str] = KLAVANS_BOYACK_ORDER


SINGLE_DOCUMENT_STYLE = OverlayStyle(height=780, axis_range=4.0)


def render_backbone_overlay(data: BackboneOverlayData, style: OverlayStyle | None = None) -> go.Figure:
    go = optional_import("plotly.graph_objects", "viz", "Rendering figures")
    style = style or OverlayStyle()
    taxonomy, mapping = data.taxonomy, data.mapping

    active = data.active_labels()
    layout = radial_layout(
        taxonomy,
        {label: mapping.primary_domain(label) for label in active},
        list(data.outer_nodes),
        radii=style.radii,
        arc_subdisciplines_deg=style.arc_subdisciplines_deg,
        arc_groups_deg=style.arc_groups_deg,
        domain_order=style.domain_order,
    )
    pos = layout.positions
    all_domains = layout.ordered_domains + layout.extra_domains
    colors = domain_colors(all_domains)
    dom_w = data.projection.domain_weights
    sub_w = data.projection.subdiscipline_weights
    max_dom = max(dom_w.values(), default=0.0)
    max_sub = max(sub_w.values(), default=0.0)
    traces: list[Any] = []

    # Domain petals: hull around each domain node and its active groups.
    for domain in all_domains:
        groups = layout.groups_by_domain.get(domain, [])
        if not groups:
            continue
        outline = petal_outline(
            ([pos[domain]] if domain in pos else []) + [pos[g] for g in groups], style.hull_margin
        )
        if not outline:
            continue
        color = colors.get(domain, NEUTRAL_GRAY)
        traces.append(
            go.Scatter(
                x=[p[0] for p in outline] + [outline[0][0]],
                y=[p[1] for p in outline] + [outline[0][1]],
                mode="lines",
                line=dict(color=hex_to_rgba(color, 0.35), width=1.2, shape="spline"),
                fill="toself",
                fillcolor=hex_to_rgba(color, 0.10),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Edges.
    scaffold = [
        (node.domain, subdiscipline_key(node.domain, sub))
        for node in taxonomy
        for sub in node.effective_subdisciplines
    ]
    anchor = [(g, mapping.primary_domain(g)) for g in active if mapping.primary_domain(g) in pos]
    hit = [
        (g, subdiscipline_key(a.domain, a.subdiscipline))
        for g in active
        for a in mapping[g]
        if subdiscipline_key(a.domain, a.subdiscipline) in pos
    ]
    overlay = [
        (outer_key(node), g) for node, g, w in data.outer_edges if w > 0 and g in pos
    ]
    for edges, line in (
        (scaffold, dict(color="#e8e8e8", width=0.5)),
        (anchor, dict(color="#cccccc", width=0.5, dash="dot")),
        (hit, dict(color=NEUTRAL_GRAY, width=1.0)),
        (overlay, dict(color=hex_to_rgba(style.hot, 0.45), width=0.9)),
    ):
        if edges:
            xs, ys = _edge_coordinates(edges, pos)
            traces.append(
                go.Scatter(x=xs, y=ys, mode="lines", line=line, hoverinfo="none", showlegend=False)
            )

    # Subdisciplines: heat-colored and sized by projected weight.
    subs = [(node.domain, sub) for node in taxonomy for sub in node.effective_subdisciplines]
    sub_weights = [sub_w.get(pair, 0.0) for pair in subs]
    traces.append(
        go.Scatter(
            x=[pos[subdiscipline_key(*pair)][0] for pair in subs],
            y=[pos[subdiscipline_key(*pair)][1] for pair in subs],
            mode="markers",
            marker=dict(
                size=[6 + 16 * (w / max_sub if max_sub > 0 else 0) for w in sub_weights],
                color=[heat_color(w, max_sub, style.heat_base, style.hot) for w in sub_weights],
                line=dict(color="white", width=0.5),
            ),
            hovertext=[
                f"<b>{sub}</b><br>parent: {domain}<br>projected weight: {w:.1f}"
                for (domain, sub), w in zip(subs, sub_weights)
            ],
            hoverinfo="text",
            name="Subdiscipline (projected weight)",
        )
    )

    # Domains: published color, opacity scales with projected weight.
    domains = layout.ordered_domains
    traces.append(
        go.Scatter(
            x=[pos[d][0] for d in domains],
            y=[pos[d][1] for d in domains],
            mode="markers+text",
            marker=dict(
                size=22,
                color=[colors.get(d, "#888") for d in domains],
                opacity=[
                    0.35 + 0.65 * (dom_w.get(d, 0.0) / max_dom if max_dom > 0 else 0)
                    for d in domains
                ],
                line=dict(color="white", width=2),
            ),
            text=[d.split(" & ")[0] if len(d) > 22 else d for d in domains],
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            hovertext=[
                f"<b>{d}</b> (domain)<br>projected weight: {dom_w.get(d, 0.0):.1f}<br>"
                f"{len(taxonomy[d].subdisciplines)} subdisciplines"
                for d in domains
            ],
            hoverinfo="text",
            name="Domain",
        )
    )

    # Groups: glow plus crisp circle, sized by weight, grouped by primary domain.
    for domain in all_domains:
        groups = layout.groups_by_domain.get(domain, [])
        if not groups:
            continue
        color = colors.get(domain, NEUTRAL_GRAY)
        weights = [data.label_weights.get(g, 0.0) for g in groups]
        sizes = [12 + math.sqrt(max(w, 1)) * 2.0 for w in weights]
        xs, ys = [pos[g][0] for g in groups], [pos[g][1] for g in groups]
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=[s * 2.0 for s in sizes], color=hex_to_rgba(color, 0.20), line=dict(width=0)),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(size=sizes, color=color, symbol="circle", line=dict(color="white", width=1.6)),
                text=[
                    f"{g} ({w:g})" if data.show_weights_in_labels else g
                    for g, w in zip(groups, weights)
                ],
                textposition=[label_position(x, y) for x, y in zip(xs, ys)],
                textfont=dict(size=9, color=color),
                hovertext=[
                    f"<b>{g}</b><br>primary domain: {domain}<br>{data.label_noun}: {w:g}"
                    for g, w in zip(groups, weights)
                ],
                hoverinfo="text",
                name=f"Group → {domain}",
            )
        )

    # Outer ring (documents).
    if data.outer_nodes:
        labels = list(data.outer_nodes)
        traces.append(
            go.Scatter(
                x=[pos[outer_key(n)][0] for n in labels],
                y=[pos[outer_key(n)][1] for n in labels],
                mode="markers",
                marker=dict(size=11, color=style.hot, symbol="circle-open", line=dict(color=style.hot, width=2)),
                hovertext=[data.outer_nodes[n] for n in labels],
                hoverinfo="text",
                name=f"Documents ({len(labels)})",
            )
        )

    title = f"{data.title}<br><sub>{data.subtitle}</sub>" if data.subtitle else data.title
    return go.Figure(
        traces,
        layout=go.Layout(
            title=title,
            showlegend=True,
            hovermode="closest",
            height=style.height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-style.axis_range, style.axis_range]),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1,
                range=[-style.axis_range, style.axis_range],
            ),
            plot_bgcolor="white",
            legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, font=dict(size=10)),
            margin=dict(l=20, r=20, t=80, b=20),
        ),
    )


def _edge_coordinates(
    edges: Sequence[tuple[str, str]], pos: dict[str, tuple[float, float]]
) -> tuple[list[float | None], list[float | None]]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    for u, v in edges:
        (ux, uy), (vx, vy) = pos.get(u, (0.0, 0.0)), pos.get(v, (0.0, 0.0))
        xs += [ux, vx, None]
        ys += [uy, vy, None]
    return xs, ys
