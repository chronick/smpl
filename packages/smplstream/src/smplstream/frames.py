"""Frame construction, validation, and the kind table (spec → *The frame*, *Frame kinds*).

Frames are plain dicts on the wire (so unknown fields survive passthrough). This module
gives typed constructors and a structural validator over those dicts.
"""

from __future__ import annotations

from typing import Any, Optional

from .ids import PROTOCOL_VERSION, mint_id

KINDS = frozenset(
    {"audio", "image", "video", "midi", "text", "vector", "marker", "feature", "control", "error"}
)

# Heavy kinds carry bytes via `hash`; small kinds inline via `data`. `vector` is size-split.
_HASH_KINDS = frozenset({"audio", "image", "video"})
_DATA_KINDS = frozenset({"text", "marker", "feature", "control", "error"})

# Inline payload ceiling (spec → *Inline payloads & size limits*).
MAX_INLINE_BYTES = 64 * 1024

# Vector inline/CAS split (spec → *Frame kinds / vector*).
VECTOR_INLINE_MAX_DIM = 64


def _attach_lineage(frame: dict, of: Optional[str], lineage: Optional[list[str]]) -> None:
    if of is not None:
        frame["of"] = of
    if lineage:
        frame["lineage"] = list(lineage)


def audio_frame(
    hash: str,
    *,
    sr: int,
    ch: int,
    dur: float,
    role: Optional[str] = None,
    of: Optional[str] = None,
    lineage: Optional[list[str]] = None,
    op: Optional[str] = None,
    op_version: Optional[str] = None,
    params: Optional[dict] = None,
    media: str = "audio/wav",
    bits: Optional[int] = None,
    fmt: Optional[str] = None,
    seq: Optional[int] = None,
) -> dict:
    meta: dict[str, Any] = {"sr": sr, "ch": ch, "dur": dur}
    if bits is not None:
        meta["bits"] = bits
    if fmt is not None:
        meta["fmt"] = fmt
    frame: dict[str, Any] = {"kind": "audio", "hash": hash, "media": media, "meta": meta}
    if role:
        frame["role"] = role
    _attach_lineage(frame, of, lineage)
    for k, v in (("op", op), ("op_version", op_version), ("params", params)):
        if v is not None:
            frame[k] = v
    if seq is not None:
        frame["seq"] = seq
    return mint_id(frame)


def text_frame(text: str, *, role: str = "caption", of: Optional[str] = None, **kw) -> dict:
    frame: dict[str, Any] = {"kind": "text", "role": role, "data": text}
    _attach_lineage(frame, of, kw.get("lineage"))
    for k in ("op", "op_version", "params"):
        if kw.get(k) is not None:
            frame[k] = kw[k]
    return mint_id(frame)


def feature_frame(data: dict, *, role: Optional[str] = None, of: Optional[str] = None, **kw) -> dict:
    frame: dict[str, Any] = {"kind": "feature", "data": data}
    if role:
        frame["role"] = role
    _attach_lineage(frame, of, kw.get("lineage"))
    for k in ("op", "op_version", "params"):
        if kw.get(k) is not None:
            frame[k] = kw[k]
    return mint_id(frame)


def marker_frame(points: list[dict], *, role: str = "onset", of: Optional[str] = None, **kw) -> dict:
    frame: dict[str, Any] = {"kind": "marker", "role": role, "data": points}
    _attach_lineage(frame, of, kw.get("lineage"))
    for k in ("op", "op_version", "params"):
        if kw.get(k) is not None:
            frame[k] = kw[k]
    return mint_id(frame)


def image_frame(hash: str, *, media: str = "image/png", role: str = "spectrogram",
                of: Optional[str] = None, **kw) -> dict:
    frame: dict[str, Any] = {"kind": "image", "hash": hash, "media": media, "role": role}
    _attach_lineage(frame, of, kw.get("lineage"))
    for k in ("op", "op_version", "params", "meta"):
        if kw.get(k) is not None:
            frame[k] = kw[k]
    return mint_id(frame)


def vector_frame(
    *,
    model: str,
    dim: int,
    dtype: str = "float32",
    data: Optional[list[float]] = None,
    hash: Optional[str] = None,
    media: str = "application/x-npy",
    role: Optional[str] = None,
    of: Optional[str] = None,
    **kw,
) -> dict:
    """A `vector` frame. dim<=64 inlines as `data`; dim>64 references CAS via `hash`."""
    if (data is None) == (hash is None):
        raise ValueError("vector frame needs exactly one of inline `data` or CAS `hash`")
    meta = {"model": model, "dim": dim, "dtype": dtype}
    frame: dict[str, Any] = {"kind": "vector", "meta": meta}
    if data is not None:
        frame["data"] = data
    else:
        frame["hash"] = hash
        frame["media"] = media
    if role:
        frame["role"] = role
    _attach_lineage(frame, of, kw.get("lineage"))
    for k in ("op", "op_version", "params"):
        if kw.get(k) is not None:
            frame[k] = kw[k]
    return mint_id(frame)


def _serialized_data_size(frame: dict) -> int:
    from . import ndjson

    return len(ndjson.dumps(frame.get("data")))


def validate_frame(frame: dict) -> list[str]:
    """Structural validation against the normative rules. Returns a list of problems."""
    problems: list[str] = []
    if frame.get("v") is None:
        problems.append("missing required field `v`")
    elif frame["v"] > PROTOCOL_VERSION:
        problems.append(f"frame v={frame['v']} exceeds supported v={PROTOCOL_VERSION} (reject)")
    kind = frame.get("kind")
    if kind is None:
        problems.append("missing required field `kind`")
    if frame.get("id") is None:
        problems.append("missing required field `id`")
    has_hash = frame.get("hash") is not None
    has_data = frame.get("data") is not None
    if has_hash and has_data:
        problems.append("`hash` and `data` are mutually exclusive (never both)")
    if has_hash and not frame.get("media"):
        problems.append("`media` is REQUIRED whenever `hash` is present")
    if has_hash:
        from .cas import HASH_RE

        if not HASH_RE.match(str(frame["hash"])):
            problems.append(f"`hash` not in canonical form: {frame['hash']!r}")
    if has_data and _serialized_data_size(frame) > MAX_INLINE_BYTES:
        problems.append(f"inline `data` exceeds {MAX_INLINE_BYTES} bytes; move to CAS")
    if kind in _HASH_KINDS and not (has_hash or has_data):
        problems.append(f"`{kind}` frame must carry a payload (`hash` or, for lazy promises, a `hash`)")
    if kind == "error":
        d = frame.get("data") or {}
        from .errors import ERROR_CODES

        if not isinstance(d, dict) or d.get("code") not in ERROR_CODES:
            problems.append("`error` frame must carry data.code from the standard enum")
        elif not d.get("message"):
            problems.append("`error` frame data MUST include a `message`")
    if kind == "vector":
        meta = frame.get("meta") or {}
        dim = meta.get("dim")
        if dim is None:
            problems.append("`vector` frame must declare meta.dim")
        elif has_data and dim > VECTOR_INLINE_MAX_DIM:
            problems.append(f"vector dim={dim} > {VECTOR_INLINE_MAX_DIM} MUST go to CAS, not inline")
    return problems


def is_kind(frame: dict, kind: str) -> bool:
    return frame.get("kind") == kind


def find_duplicate_ids(frames: list[dict]) -> list[str]:
    """Return ids carried by two or more DISTINCT frames (a normative id collision).

    Identical repeats of the same frame (content-id dedup) are not collisions.
    """
    from .conformance import check_id_collisions

    seen = set()
    out = []
    for problem in check_id_collisions(frames):
        # problem text ends with the colliding id
        fid = problem.rsplit(" ", 1)[-1]
        if fid not in seen:
            seen.add(fid)
            out.append(fid)
    return out
