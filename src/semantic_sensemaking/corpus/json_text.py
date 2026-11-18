"""Extract prose from transcript-shaped JSON."""

from __future__ import annotations

import json
from typing import Any


def extract_json_text(data: Any) -> str:
    """Pull the most informative text field from a parsed JSON structure.

    Checked in order: a top-level ``text`` longer than 100 characters (meeting
    recording shape), ``chunks[].text``, ``speakers[].text``, then ``content``.
    Anything else is serialized back to JSON so it can still be scored.
    """
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return json.dumps(data)
    if isinstance(data.get("text"), str) and len(data["text"]) > 100:
        return data["text"]
    for key in ("chunks", "speakers"):
        items = data.get(key)
        if isinstance(items, list) and items:
            parts = [str(item.get("text", "")) for item in items if isinstance(item, dict)]
            if any(parts):
                return " ".join(parts)
    if isinstance(data.get("content"), str):
        return data["content"]
    return json.dumps(data)
