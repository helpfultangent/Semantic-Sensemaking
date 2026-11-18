"""Test oracle: the SUBSIDE Backbone Overlay prototype's analysis, ported verbatim.

Cells 8, 10, 12 and 20 of ``3-MediaMisinfo_backbone_overlay.ipynb``. Notebook
globals became function parameters and print statements were removed; the
logic is otherwise unchanged, bugs included. Library code must never import
this module — it exists so parity tests can show exactly where the library
matches the prototype and where deliberate fixes diverge from it.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ── Cell 10 ──────────────────────────────────────────────────────────────
def _extract_json_text(data):
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return json.dumps(data)
    if isinstance(data.get("text"), str) and len(data["text"]) > 100:
        return data["text"]
    if isinstance(data.get("chunks"), list) and data["chunks"]:
        parts = [str(c.get("text", "")) for c in data["chunks"] if isinstance(c, dict)]
        if any(parts):
            return " ".join(parts)
    if isinstance(data.get("speakers"), list) and data["speakers"]:
        parts = [str(s.get("text", "")) for s in data["speakers"] if isinstance(s, dict)]
        if any(parts):
            return " ".join(parts)
    if isinstance(data.get("content"), str):
        return data["content"]
    return json.dumps(data)


def load_overlay_file(filepath: Path):
    suf = filepath.suffix.lower()
    try:
        if suf in (".txt", ".md"):
            return filepath.read_text(encoding="utf-8", errors="replace"), suf[1:]
        if suf == ".json":
            with open(filepath, encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
            return _extract_json_text(data), "json"
        if suf == ".docx":
            from docx import Document as _Docx

            doc = _Docx(str(filepath))
            return "\n".join(p.text for p in doc.paragraphs), "docx"
        if suf == ".pdf":
            from pypdf import PdfReader as _PdfReader

            reader = _PdfReader(str(filepath))
            return "\n".join((p.extract_text() or "") for p in reader.pages), "pdf"
        return None, f"unsupported format ({suf})"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def load_overlay_corpus(path: Path, glob_patterns=("*",), max_files=200, min_chars=200):
    docs = {}
    skipped = []
    candidate_files = []
    if path.is_file():
        candidate_files = [path]
    elif path.is_dir():
        for pat in glob_patterns:
            candidate_files.extend(sorted(path.glob(pat)))
        seen = set()
        uniq = []
        for f in candidate_files:
            if f not in seen and f.is_file():
                seen.add(f)
                uniq.append(f)
        candidate_files = uniq[:max_files]
    else:
        return docs, [(str(path), "path does not exist")]
    for f in candidate_files:
        if f.name.startswith(".") or f.name.startswith("~$"):
            continue
        if "checkpoint" in f.name.lower():
            continue
        text, kind = load_overlay_file(f)
        if text is None:
            skipped.append((f.name, kind))
            continue
        if len(text.strip()) < min_chars:
            skipped.append((f.name, f"too short ({len(text.strip())} chars)"))
            continue
        docs[f.stem] = text
    return docs, skipped


# ── Cell 8 ───────────────────────────────────────────────────────────────
def map_terms_to_backbone(terms, backbone, top_n=3):
    terms_l = [str(t).lower() for t in terms]
    hits = []
    for domain, node in backbone.items():
        subs = node.get("subdisciplines") or ["General"]
        node_terms = node.get("terms") or [domain.lower()]
        for sub in subs:
            score = sum(1 for t in terms_l if any(str(kw).lower() in t for kw in node_terms))
            if score > 0:
                hits.append((domain, sub, score))
    if not hits:
        return [("Uncategorized", "General", 0)]
    return sorted(hits, key=lambda x: -x[2])[:top_n]


def build_group_mapping_df(keyword_groups, backbone):
    rows = []
    for name, terms in keyword_groups.items():
        mapping = map_terms_to_backbone(terms, backbone)
        rows.append(
            {"group": name, "terms": terms, "primary_domain": mapping[0][0], "backbone_mapping": mapping}
        )
    return pd.DataFrame(rows)


# ── Cell 12 ──────────────────────────────────────────────────────────────
def count_group_in_text(text, group_terms):
    text_l = text.lower()
    total = 0
    for term in group_terms:
        t = term.lower()
        if " " in t or "-" in t:
            total += text_l.count(t)
        else:
            total += len(re.findall(rf"\b{re.escape(t)}\b", text_l))
    return total


def project_overlay(overlay_docs, keyword_groups, group_mapping_df):
    rows = []
    for doc_name, text in overlay_docs.items():
        n_words = len(text.split())
        for group, terms in keyword_groups.items():
            c = count_group_in_text(text, terms)
            rows.append(
                {
                    "doc": doc_name,
                    "group": group,
                    "count": c,
                    "doc_n_words": n_words,
                    "per_1k_words": round(1000 * c / max(n_words, 1), 2),
                }
            )
    overlay_doc_group_df = pd.DataFrame(rows)
    overlay_group_totals = (
        overlay_doc_group_df.groupby("group")["count"].sum().reset_index().sort_values("count", ascending=False)
    )
    domain_weights = defaultdict(float)
    sub_weights = defaultdict(float)
    for _, grow in group_mapping_df.iterrows():
        g = grow["group"]
        mapping = grow["backbone_mapping"]
        g_count = float(overlay_group_totals.set_index("group")["count"].get(g, 0))
        if g_count <= 0 or not mapping:
            continue
        total_score = sum(s for _, _, s in mapping) or 1.0
        for domain, sub, score in mapping:
            w = g_count * (score / total_score)
            domain_weights[domain] += w
            sub_weights[(domain, sub)] += w
    overlay_domain_totals = (
        pd.DataFrame([{"domain": d, "weight": w} for d, w in domain_weights.items()])
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "doc_group": overlay_doc_group_df,
        "group_totals": overlay_group_totals,
        "domain_weights": dict(domain_weights),
        "sub_weights": dict(sub_weights),
        "domain_totals": overlay_domain_totals,
    }


# ── Cell 20 ──────────────────────────────────────────────────────────────
def comparison_table(group_mapping_df, overlay_domain_totals):
    base_dist = (
        group_mapping_df.groupby("primary_domain").size().reset_index(name="base_n_groups")
        .rename(columns={"primary_domain": "domain"})
    )
    ov_dist = overlay_domain_totals.copy()
    ov_dist["overlay_weight"] = ov_dist["weight"]
    ov_dist = ov_dist.drop(columns=["weight"])
    base_dist["base_pct"] = 100 * base_dist["base_n_groups"] / base_dist["base_n_groups"].sum()
    ov_total = ov_dist["overlay_weight"].sum() or 1.0
    ov_dist["overlay_pct"] = 100 * ov_dist["overlay_weight"] / ov_total
    comparison = pd.merge(base_dist, ov_dist, on="domain", how="outer").fillna(0).sort_values(
        "overlay_pct", ascending=False
    )
    comparison["delta_pct"] = comparison["overlay_pct"] - comparison["base_pct"]
    return comparison


# ── Inputs in the prototype's dict shapes ────────────────────────────────
def keyword_groups_from_file(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text())
    return {g["name"]: g["terms"] for g in data["groups"]}


def backbone_from_file(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {
        d["domain"]: {"subdisciplines": d["subdisciplines"], "terms": d["terms"]} for d in data["domains"]
    }


def run_prototype(corpus_dir: Path, lexicon_file: Path, taxonomy_file: Path) -> dict:
    """The prototype's §2–§6 analysis end to end."""
    keyword_groups = keyword_groups_from_file(lexicon_file)
    backbone = backbone_from_file(taxonomy_file)
    docs, skipped = load_overlay_corpus(corpus_dir)
    mapping_df = build_group_mapping_df(keyword_groups, backbone)
    projected = project_overlay(docs, keyword_groups, mapping_df)
    return {
        "docs": docs,
        "skipped": skipped,
        "group_mapping_df": mapping_df,
        **projected,
        "comparison": comparison_table(mapping_df, projected["domain_totals"]),
    }
