"""Frame id assignment (spec → *Id assignment*, collision-proof by construction).

A frame's ``id`` MUST be globally unique without coordination so independently-numbered
streams merge without reference corruption. Producers mint a **content-derived** token
(``blake3:`` over the frame's defining fields — preferred, so the *same* frame from two
pipelines shares one id and deduplicates) or a random token. A per-stream sequential
counter (``f1``, ``f2``…) is the collision trap and is forbidden.
"""

from __future__ import annotations

import secrets

from blake3 import blake3

from .memo import canonical_json

# Fields that define a frame's identity. seq/consumed/meta/media/id/v are excluded:
# they are positional / derivable / structural, not part of "which frame this is".
_DEFINING_FIELDS = (
    "kind",
    "role",
    "of",
    "lineage",
    "op",
    "op_version",
    "params",
    "hash",
    "data",
)

PROTOCOL_VERSION = 1


def content_id(frame: dict) -> str:
    """Content-derived id over a frame's defining fields. Stable + dedup-friendly."""
    defining = {k: frame[k] for k in _DEFINING_FIELDS if k in frame and frame[k] is not None}
    digest = blake3(canonical_json(defining)).hexdigest()
    return "blake3:" + digest


def random_id() -> str:
    """A random frame id (use when a frame must be unique even if structurally identical)."""
    return "rand:" + secrets.token_hex(32)


def mint_id(frame: dict, *, random: bool = False) -> dict:
    """Stamp ``v`` and a fresh ``id`` onto a frame in place, returning it.

    Content-derived by default; ``random=True`` forces a random token. An ``id`` already
    present is preserved (passthrough frames keep their inbound id verbatim).
    """
    frame.setdefault("v", PROTOCOL_VERSION)
    if "id" not in frame or frame["id"] is None:
        frame["id"] = random_id() if random else content_id(frame)
    return frame
