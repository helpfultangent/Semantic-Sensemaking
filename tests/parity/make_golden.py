"""Regenerate ``golden_sample_corpus.json`` from the prototype oracle.

Run only when the sample corpus or built-in data changes on purpose:

    python tests/parity/make_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import prototype_reference as proto  # noqa: E402

from semantic_sensemaking import sample_corpus_path  # noqa: E402
from semantic_sensemaking.data import builtin_path  # noqa: E402


def main() -> None:
    result = proto.run_prototype(
        sample_corpus_path(),
        builtin_path("lexicon_subside_v1.json"),
        builtin_path("taxonomy_subside_v1.json"),
    )
    golden = {
        "source": "tests/parity/prototype_reference.py on the built-in sample corpus",
        "group_totals": {
            g: int(c) for g, c in zip(result["group_totals"]["group"], result["group_totals"]["count"])
        },
        "domain_weights": result["domain_weights"],
        "subdiscipline_weights": {f"{d} :: {s}": w for (d, s), w in result["sub_weights"].items()},
    }
    out = Path(__file__).with_name("golden_sample_corpus.json")
    out.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
