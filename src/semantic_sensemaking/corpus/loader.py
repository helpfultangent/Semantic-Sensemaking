"""Load a file or directory into a :class:`~semantic_sensemaking.core.types.Corpus`."""

from __future__ import annotations

import fnmatch
import logging
from collections import Counter
from pathlib import Path

from semantic_sensemaking.core.config import CorpusConfig
from semantic_sensemaking.core.types import Corpus, Document, SkippedFile
from semantic_sensemaking.corpus.readers import read_document
from semantic_sensemaking.errors import CorpusError

log = logging.getLogger(__name__)


def load_corpus(path: str | Path, config: CorpusConfig | None = None) -> Corpus:
    """Read every supported file under ``path`` into a corpus.

    ``path`` may be a single file or a directory. Hidden files, Office lock files
    and notebook checkpoints are ignored (``config.ignore_patterns``); unreadable
    or too-short files are reported in ``Corpus.skipped``.

    ``doc_id`` is the path relative to the corpus root, so two files that share a
    stem (``a.txt`` and ``a.json``) no longer overwrite each other as they did in
    the prototype. ``label`` stays the stem unless stems collide.
    """
    config = config or CorpusConfig()
    path = Path(path)
    if path.is_file():
        root, candidates = path.parent, [path]
    elif path.is_dir():
        root, candidates = path, _discover(path, config)
    else:
        raise CorpusError(f"Corpus path does not exist: {path}")

    skipped: list[SkippedFile] = []
    if len(candidates) > config.max_files:
        log.warning(
            "Corpus has %d files; reading the first %d (corpus.max_files)",
            len(candidates),
            config.max_files,
        )
        skipped.extend(
            SkippedFile(_rel(f, root), f"exceeds corpus.max_files ({config.max_files})")
            for f in candidates[config.max_files :]
        )
        candidates = candidates[: config.max_files]

    loaded: list[tuple[Path, str, str]] = []
    for file in candidates:
        text, kind = read_document(file, encoding=config.encoding)
        if text is None:
            skipped.append(SkippedFile(_rel(file, root), kind))
            continue
        n_chars = len(text.strip())
        if n_chars < config.min_chars:
            skipped.append(SkippedFile(_rel(file, root), f"too short ({n_chars} chars)"))
            continue
        loaded.append((file, text, kind))

    stem_counts = Counter(f.stem for f, _, _ in loaded)
    documents = tuple(
        Document(
            doc_id=_rel(f, root),
            text=text,
            label=f.stem if stem_counts[f.stem] == 1 else _rel(f, root),
            source_kind=kind,
            source_path=f,
        )
        for f, text, kind in loaded
    )
    log.info("Loaded %d document(s) from %s; skipped %d", len(documents), path, len(skipped))
    return Corpus(documents, tuple(skipped), root)


def _discover(root: Path, config: CorpusConfig) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in config.glob_patterns:
        matches = root.rglob(pattern) if config.recursive else root.glob(pattern)
        for f in sorted(matches):
            if f in seen or not f.is_file():
                continue
            seen.add(f)
            if _is_ignored(f.relative_to(root), config.ignore_patterns):
                log.debug("Ignoring %s", f)
                continue
            files.append(f)
    return files


def _is_ignored(relative: Path, patterns: list[str]) -> bool:
    return any(
        fnmatch.fnmatch(part.lower(), pattern.lower())
        for part in relative.parts
        for pattern in patterns
    )


def _rel(file: Path, root: Path) -> str:
    try:
        return file.relative_to(root).as_posix()
    except ValueError:
        return file.name
