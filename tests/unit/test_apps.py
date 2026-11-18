"""Tapis app definitions stay in sync with the configuration they set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_sensemaking.core.config import RunConfig
from semantic_sensemaking.workflows import WORKFLOWS

APPS = sorted((Path(__file__).parents[2] / "apps").glob("*/app.json"))


def test_every_workflow_that_should_have_an_app_has_one():
    assert {p.parent.name for p in APPS} >= {"dictionary", "narrative"}


@pytest.mark.parametrize("path", APPS, ids=lambda p: p.parent.name)
def test_app_env_vars_are_valid_settings(path, monkeypatch):
    app = json.loads(path.read_text())
    env = {e["key"]: e["value"] for e in app["jobAttributes"]["parameterSet"]["envVariables"]}
    assert env["SSM_WORKFLOW"] == path.parent.name in WORKFLOWS
    assert app["id"] == f"semantic-sensemaking-{path.parent.name}"
    for key, value in env.items():
        assert key.startswith("SSM_"), key
        monkeypatch.setenv(key, value)
    # Unknown keys or invalid values raise ConfigError (extra="forbid").
    config = RunConfig.load()
    assert config.workflow == path.parent.name
    assert {f["targetPath"] for f in app["jobAttributes"]["fileInputs"]} >= {"corpus", "job.yaml"}
