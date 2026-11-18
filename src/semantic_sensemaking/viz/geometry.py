"""Radial layout and polygon helpers for backbone figures (no plotting dependency)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from semantic_sensemaking.core.types import Taxonomy
from semantic_sensemaking.viz.palette import KLAVANS_BOYACK_ORDER, order_domains

Point = tuple[float, float]


@dataclass(frozen=True)
class Radii:
    domain: float = 1.0
    subdiscipline: float = 1.7
    group: float = 2.7
    outer: float = 3.6


def subdiscipline_key(domain: str, subdiscipline: str) -> str:
    return f"{domain} :: {subdiscipline}"


def outer_key(label: str) -> str:
    return f"OVERLAY::{label}"


def fan(base: float, n: int, arc: float, r: float) -> list[Point]:
    """``n`` points on radius ``r`` spread over ``arc`` radians centred on angle ``base``."""
    if n <= 0:
        return []
    if n == 1:
        return [(r * math.cos(base), r * math.sin(base))]
    return [
        (
            r * math.cos(base - arc / 2 + (j / (n - 1)) * arc),
            r * math.sin(base - arc / 2 + (j / (n - 1)) * arc),
        )
        for j in range(n)
    ]


def convex_hull(points: Sequence[Point]) -> list[Point]:
    """Andrew's monotone chain; counter-clockwise hull (fewer than 3 points returned as-is)."""
    pts = sorted(set(points))
    if len(pts) < 2:
        return pts

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def inflate_hull(hull: Sequence[Point], margin: float = 0.20) -> list[Point]:
    """Push each vertex ``margin`` further from the hull's centroid."""
    if len(hull) < 2:
        return list(hull)
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    out = []
    for x, y in hull:
        dx, dy = x - cx, y - cy
        dist = math.hypot(dx, dy) or 1.0
        out.append((x + dx / dist * margin, y + dy / dist * margin))
    return out


def petal_outline(points: Sequence[Point], margin: float = 0.22, half_width: float = 0.15) -> list[Point]:
    """Inflated hull around ``points``; a two-point hull is widened into a quadrilateral.

    Returns an empty list when there are fewer than two distinct points.
    """
    if len(set(points)) < 2:
        return []
    hull = inflate_hull(convex_hull(points), margin)
    if len(hull) == 2:
        (x1, y1), (x2, y2) = hull
        length = math.hypot(x2 - x1, y2 - y1) or 1.0
        nx, ny = -(y2 - y1) / length * half_width, (x2 - x1) / length * half_width
        hull = [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)]
    return hull


def label_position(x: float, y: float) -> str:
    """Plotly ``textposition`` that points a label away from the figure centre."""
    if x >= 0.3 and y >= 0.3:
        return "top right"
    if x <= -0.3 and y >= 0.3:
        return "top left"
    if x <= -0.3 and y <= -0.3:
        return "bottom left"
    if x >= 0.3 and y <= -0.3:
        return "bottom right"
    if x >= 0.3:
        return "middle right"
    if x <= -0.3:
        return "middle left"
    if y >= 0.3:
        return "top center"
    return "bottom center"


@dataclass
class RadialLayout:
    positions: dict[str, Point]
    domain_angle: dict[str, float]
    ordered_domains: list[str]
    extra_domains: list[str]
    groups_by_domain: dict[str, list[str]] = field(default_factory=dict)


def radial_layout(
    taxonomy: Taxonomy,
    group_domains: Mapping[str, str],
    outer_nodes: Sequence[str] = (),
    *,
    radii: Radii | None = None,
    arc_subdisciplines_deg: float = 24.0,
    arc_groups_deg: float = 20.0,
    domain_order: Sequence[str] = KLAVANS_BOYACK_ORDER,
) -> RadialLayout:
    """Four concentric rings: domains, subdisciplines, groups, outer (document) nodes.

    ``group_domains`` maps each group to draw onto its primary domain. Groups whose
    domain is not in the taxonomy (``Uncategorized``) are fanned near the top.
    """
    radii = radii or Radii()
    ordered = order_domains(taxonomy.domains(), domain_order)
    pos: dict[str, Point] = {}
    angle: dict[str, float] = {}
    for i, domain in enumerate(ordered):
        theta = -math.pi / 2 - 2 * math.pi * i / max(len(ordered), 1)
        angle[domain] = theta
        pos[domain] = (radii.domain * math.cos(theta), radii.domain * math.sin(theta))

    arc_subs = math.radians(arc_subdisciplines_deg)
    for domain in ordered:
        subs = taxonomy[domain].effective_subdisciplines
        for sub, xy in zip(subs, fan(angle[domain], len(subs), arc_subs, radii.subdiscipline)):
            pos[subdiscipline_key(domain, sub)] = xy

    groups_by_domain: dict[str, list[str]] = {}
    for group, domain in group_domains.items():
        groups_by_domain.setdefault(domain, []).append(group)

    extra = [d for d in groups_by_domain if d not in angle]
    for k, domain in enumerate(extra):
        angle[domain] = math.pi / 2 + (k - (len(extra) - 1) / 2) * math.radians(10)

    for domain, groups in groups_by_domain.items():
        n = len(groups)
        coords = fan(angle[domain], n, max(math.radians(arc_groups_deg), math.radians(8 * n)), radii.group)
        if n > 3:  # stagger alternate groups outward so labels don't collide
            for j in range(1, n, 2):
                x, y = coords[j]
                scale = (math.hypot(x, y) + 0.35) / math.hypot(x, y)
                coords[j] = (x * scale, y * scale)
        pos.update(zip(groups, coords))

    for j, label in enumerate(outer_nodes):
        theta = -math.pi / 2 - 2 * math.pi * j / max(len(outer_nodes), 1)
        pos[outer_key(label)] = (radii.outer * math.cos(theta), radii.outer * math.sin(theta))

    return RadialLayout(pos, angle, ordered, extra, groups_by_domain)
