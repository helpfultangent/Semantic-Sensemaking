"""Pieces: the workflows cut into steps a workflow engine can run one at a time.

Each piece reads its inputs by location, calls the tool clusters, and writes its
outputs back. See ``README.md`` in this package.
"""

from semantic_sensemaking.pieces.catalog import SPECS, PieceSpec, registration, stored_source
from semantic_sensemaking.pieces.entrypoints import PIECES, run_piece
from semantic_sensemaking.pieces.storage import Storage, StorageError

__all__ = [
    "PIECES",
    "SPECS",
    "PieceSpec",
    "Storage",
    "StorageError",
    "registration",
    "run_piece",
    "stored_source",
]
