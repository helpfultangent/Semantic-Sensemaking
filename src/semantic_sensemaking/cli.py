"""``semantic-sensemaking`` command line: one entrypoint, one subcommand per workflow.

Exit codes: 0 success, 1 runtime failure, 2 invalid configuration or a workflow
that isn't implemented yet.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_sensemaking._version import __version__
from semantic_sensemaking.core.config import RunConfig
from semantic_sensemaking.errors import ConfigError, SemanticSensemakingError
from semantic_sensemaking.workflows import WORKFLOWS, get_workflow

log = logging.getLogger("semantic_sensemaking")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-sensemaking",
        description="Semantic sensemaking workflows for local machines and TACC.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    for name in WORKFLOWS:
        aliases = ["formulate"] if name == "formulation" else []
        workflow = commands.add_parser(
            name, aliases=aliases, help=get_workflow(name).DESCRIPTION
        )
        workflow.set_defaults(workflow=name)
        _add_run_arguments(workflow)

    config = commands.add_parser("config", help="Inspect configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    show = config_commands.add_parser("show", help="Print the fully resolved configuration")
    show.add_argument("--workflow", choices=list(WORKFLOWS), default=None)
    _add_run_arguments(show)

    schema = commands.add_parser("schema", help="Write JSON Schemas for input and job files")
    schema.add_argument("output_dir", type=Path, help="Directory to write *.schema.json into")
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="YAML job configuration file")
    parser.add_argument("--corpus", type=Path, help="File or directory of documents")
    parser.add_argument(
        "--sample-corpus", action="store_true", help="Use the built-in synthetic sample corpus"
    )
    parser.add_argument("--lexicon", type=Path, help="Lexicon (dictionary) JSON or CSV file")
    parser.add_argument("--taxonomy", type=Path, help="Taxonomy (science backbone) JSON file")
    parser.add_argument("--taxonomy-layer", help="Layer to use from a layered backbone file")
    parser.add_argument("--output-dir", type=Path, help="Directory that will hold the run directory")
    parser.add_argument("--run-id", help="Run directory name (default: <workflow>_<timestamp>)")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any setting, e.g. --set scoring.match_mode=longest_match (repeatable)",
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def config_from_args(args: argparse.Namespace, workflow: str | None) -> RunConfig:
    overrides: dict[str, Any] = {}
    if workflow:
        overrides["workflow"] = workflow
    inputs = {
        key: value
        for key, value in {
            "corpus": args.corpus,
            "lexicon": args.lexicon,
            "taxonomy": args.taxonomy,
            "taxonomy_layer": args.taxonomy_layer,
            "use_sample_corpus": True if args.sample_corpus else None,
        }.items()
        if value is not None
    }
    if inputs:
        overrides["inputs"] = inputs
    output = {k: v for k, v in {"dir": args.output_dir, "run_id": args.run_id}.items() if v is not None}
    if output:
        overrides["output"] = output
    if args.log_level:
        overrides["log_level"] = args.log_level
    return RunConfig.load(args.config, overrides=overrides, set_values=args.set_values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "schema":
            return _write_schemas(args.output_dir)
        if args.command == "config":
            print(config_from_args(args, args.workflow).to_yaml(), end="")
            return 0

        config = config_from_args(args, args.workflow)
        logging.basicConfig(level=config.log_level, format="%(levelname)s %(name)s: %(message)s")
        outcome = get_workflow(config.workflow).run(config)
        print(f"Run complete: {outcome.context.run_dir}")
        return 0
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SemanticSensemakingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _write_schemas(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, schema in schema_documents().items():
        path = output_dir / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        print(path)
    return 0


def schema_documents() -> dict[str, dict[str, Any]]:
    from semantic_sensemaking.lexicon.io import lexicon_json_schema
    from semantic_sensemaking.taxonomy.io import taxonomy_json_schema

    return {
        "lexicon": lexicon_json_schema(),
        "taxonomy": taxonomy_json_schema(),
        "job-config": RunConfig.model_json_schema(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
