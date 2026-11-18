"""Writers shared by every workflow: tables, Markdown summaries, run manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from semantic_sensemaking._version import __version__
from semantic_sensemaking.core.context import RunContext


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """CSV with a UTF-8 BOM so Excel opens non-ASCII labels correctly."""
    path = Path(path)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def markdown_table(frame: pd.DataFrame, *, floatfmt: str = ".1f") -> str:
    if frame.empty:
        return "_(none)_"
    return frame.to_markdown(index=False, floatfmt=floatfmt)


def render_summary(
    title: str, facts: Sequence[tuple[str, Any]], sections: Sequence[tuple[str, str]]
) -> str:
    lines = [f"# {title}", ""]
    lines += [f"- **{label}**: {value}" for label, value in facts]
    for heading, body in sections:
        lines += ["", f"## {heading}", "", body]
    return "\n".join(lines) + "\n"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    ctx: RunContext,
    *,
    config: Mapping[str, Any],
    inputs: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> Path:
    """Record what ran, on what inputs, with which settings, producing which files."""
    path = ctx.artifact("manifest.json")
    manifest = {
        "workflow": ctx.workflow,
        "run_id": ctx.run_id,
        "started_at": ctx.started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "semantic_sensemaking": __version__,
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "inputs": dict(inputs),
        "stats": dict(stats),
        "config": dict(config),
        "artifacts": ctx.relative_artifacts(),
    }
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path


__all__ = ["file_sha256", "markdown_table", "render_summary", "write_csv", "write_manifest"]
