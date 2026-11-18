"""Colors and domain ordering for backbone figures."""

from __future__ import annotations

from collections.abc import Sequence

# Klavans/Boyack circular ordering of the UCSD map-of-science domains.
KLAVANS_BOYACK_ORDER: tuple[str, ...] = (
    "Humanities",
    "Social Sciences",
    "Electrical Engineering & Computer Science",
    "Math & Physics",
    "Chemistry",
    "Chemical, Mechanical, & Civil Engineering",
    "Earth Sciences",
    "Biology",
    "Biotechnology",
    "Infectious Disease",
    "Medical Specialties",
    "Health Professionals",
    "Brain Research",
)

# Published UCSD map-of-science domain colors.
UCSD_HEX: dict[str, str] = {
    "Math & Physics": "#7570B3",
    "Chemistry": "#1F78B4",
    "Earth Sciences": "#B15928",
    "Biology": "#33A02C",
    "Biotechnology": "#FB9A99",
    "Infectious Disease": "#A6CEE3",
    "Medical Specialties": "#E31A1C",
    "Health Professionals": "#FDBF6F",
    "Brain Research": "#FF7F00",
    "Electrical Engineering & Computer Science": "#CAB2D6",
    "Chemical, Mechanical, & Civil Engineering": "#6A3D9A",
    "Social Sciences": "#FFC83D",
    "Humanities": "#FFE066",
}

TACC_ORANGE = "#BF5700"
NEUTRAL_GRAY = "#9a9a9a"
HEAT_BASE = "#f5f5f5"


def hex_to_rgba(color: str, alpha: float) -> str:
    """Convert ``#rgb``, ``#rrggbb``, or ``rgb(r, g, b)`` to an ``rgba(...)`` string."""
    s = str(color).strip()
    if s.startswith("rgb"):
        inner = s[s.index("(") + 1 : s.rindex(")")]
        r, g, b = (int(float(p.strip())) for p in inner.split(",")[:3])
        return f"rgba({r},{g},{b},{alpha})"
    h = s.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def heat_color(weight: float, vmax: float, base: str = HEAT_BASE, hot: str = TACC_ORANGE) -> str:
    """Linear interpolation from ``base`` (weight 0) to ``hot`` (weight >= vmax)."""
    t = min(1.0, weight / vmax) if vmax > 0 else 0.0
    a, b = _hex_rgb(base), _hex_rgb(hot)
    return "#{:02X}{:02X}{:02X}".format(*(int(a[k] + (b[k] - a[k]) * t) for k in range(3)))


def order_domains(
    domains: Sequence[str], preferred: Sequence[str] = KLAVANS_BOYACK_ORDER
) -> list[str]:
    """Domains in ``preferred`` order first, then any others in their given order."""
    present = set(domains)
    return [d for d in preferred if d in present] + [d for d in domains if d not in preferred]


def domain_colors(domains: Sequence[str]) -> dict[str, str]:
    """UCSD colors where published, a qualitative palette otherwise."""
    from plotly.colors import qualitative

    fallback = qualitative.Bold + qualitative.Vivid
    colors = {
        d: UCSD_HEX.get(d, fallback[i % len(fallback)]) for i, d in enumerate(domains)
    }
    colors.setdefault("Uncategorized", NEUTRAL_GRAY)
    return colors


def _hex_rgb(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
