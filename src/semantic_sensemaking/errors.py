"""Exception hierarchy shared by every tool cluster.

Everything the library raises on purpose derives from
:class:`SemanticSensemakingError`, so CLI entrypoints and Tapis jobs can
separate "the user's input or environment is wrong" (clean message, non-zero
exit) from genuine bugs (traceback).
"""

from __future__ import annotations

import importlib
from types import ModuleType


class SemanticSensemakingError(Exception):
    """Base class for errors raised deliberately by this library."""


class ConfigError(SemanticSensemakingError):
    """A run configuration is missing a value or has an invalid one."""


class InputFormatError(SemanticSensemakingError, ValueError):
    """A lexicon, taxonomy, or other structured input file is malformed."""


class CorpusError(SemanticSensemakingError):
    """A corpus could not be located or loaded."""


class DocumentLookupError(CorpusError, LookupError):
    """A document spec matched no document, or more than one."""


class BackendError(SemanticSensemakingError):
    """An embedding or chat backend failed after exhausting retries."""


class MissingExtraError(SemanticSensemakingError, ImportError):
    """An optional dependency needed for the requested feature is not installed."""

    def __init__(self, module: str, extra: str, purpose: str | None = None) -> None:
        self.module = module
        self.extra = extra
        what = purpose or "This feature"
        super().__init__(
            f"{what} requires the optional dependency '{module}'. "
            f"Install it with: pip install 'semantic-sensemaking[{extra}]'"
        )


def optional_import(module: str, extra: str, purpose: str | None = None) -> ModuleType:
    """Import ``module`` or raise :class:`MissingExtraError` naming the pip extra.

    Replaces the prototype's runtime ``pip install``: compute nodes are often
    network-restricted, so a mid-job install hangs instead of failing clearly.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(module, extra, purpose) from exc
