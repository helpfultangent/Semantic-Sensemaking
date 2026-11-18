from __future__ import annotations

from pathlib import Path

import pytest

from semantic_sensemaking.core.config import MatchMode, RunConfig, parse_set_values
from semantic_sensemaking.errors import ConfigError


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "job.yaml"
    path.write_text(text)
    return path


def test_defaults_are_valid():
    config = RunConfig.load()
    assert config.workflow == "dictionary"
    assert config.scoring.match_mode is MatchMode.AUTO
    assert config.corpus.max_files == 200


def test_precedence_yaml_then_env_then_overrides(tmp_path, monkeypatch):
    path = write_yaml(tmp_path, "corpus:\n  max_files: 10\n  min_chars: 50\nscoring:\n  top_n_assignments: 2\n")
    assert RunConfig.load(path).corpus.max_files == 10

    monkeypatch.setenv("SSM_CORPUS__MAX_FILES", "20")
    config = RunConfig.load(path)
    assert (config.corpus.max_files, config.corpus.min_chars) == (20, 50)

    config = RunConfig.load(path, overrides={"corpus": {"max_files": 30}}, set_values=["scoring.top_n_assignments=5"])
    assert (config.corpus.max_files, config.corpus.min_chars) == (30, 50)
    assert config.scoring.top_n_assignments == 5


def test_env_lists_and_enums(monkeypatch):
    monkeypatch.setenv("SSM_CORPUS__GLOB_PATTERNS", '["*.json", "*.txt"]')
    monkeypatch.setenv("SSM_SCORING__MATCH_MODE", "longest_match")
    config = RunConfig.load()
    assert config.corpus.glob_patterns == ["*.json", "*.txt"]
    assert config.scoring.match_mode is MatchMode.LONGEST_MATCH


def test_unknown_keys_fail_loudly(tmp_path):
    path = write_yaml(tmp_path, "scoring:\n  matchmode: auto\n")
    with pytest.raises(ConfigError, match="scoring.matchmode"):
        RunConfig.load(path)
    with pytest.raises(ConfigError, match="corpus.max_files"):
        RunConfig.load(set_values=["corpus.max_files=zero"])


def test_missing_config_file():
    with pytest.raises(ConfigError, match="not found"):
        RunConfig.load("does-not-exist.yaml")


def test_parse_set_values_types():
    assert parse_set_values(["a.b=3", "a.c=true", "d=[1, 2]", "e="]) == {
        "a": {"b": 3, "c": True}, "d": [1, 2], "e": None,
    }
    with pytest.raises(ConfigError):
        parse_set_values(["novalue"])


def test_yaml_round_trip(tmp_path):
    config = RunConfig.load(set_values=["scoring.mapping_method=subdiscipline_aware"])
    path = write_yaml(tmp_path, config.to_yaml())
    assert RunConfig.load(path) == config
