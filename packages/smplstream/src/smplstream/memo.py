"""Memoization (spec → *Memoization*, NORMATIVE).

Every *cacheable* op is a pure function of its inputs, implementation version, and
environment:

    memo_key = blake3( op || op_version || sorted(input_hashes) || canonicalize(params) || env_fp )

- ``op_version`` is bumped on ANY behavior change; for ML ops it MUST incorporate the
  weights identity (model blake3 / registry id+version), not just a friendly name.
- ``env_fingerprint`` pins shell-out tool versions (``sox``/``ffmpeg``/SRC quality);
  empty for pure-Python deterministic ops.
- Non-deterministic ops declare ``cacheable: false`` and skip memoization.

``canonicalize(params)`` makes two spellings of the same request one key: sorted keys,
fixed canonical number form (shortest round-trippable decimal; ``6.0`` ≡ ``6``), set-valued
params sorted while sequence-valued params keep order. Callers MUST fill omitted params
from the op's declared defaults *before* hashing (don't drop defaults — that couples the
key to the live default table).
"""

from __future__ import annotations

import struct
import subprocess
from typing import Any, Iterable

from blake3 import blake3

_SEP = b"\x00"  # component separator between the FIXED-ARITY fields only (op, version, …)


def _canonical_number(x: float | int) -> Any:
    """Fixed canonical form: integral floats collapse to int; floats use shortest round-trip."""
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            # NaN/inf have no canonical decimal; render as a stable token.
            return f"__float__:{x!r}"
        if x.is_integer():
            return int(x)
        # Python's repr is the shortest round-trippable decimal for float since 3.1.
        return float(repr(x))
    return x


def _canonicalize(obj: Any, set_keys: frozenset[str], _path: str = "") -> Any:
    if isinstance(obj, dict):
        out = {}
        for k in sorted(obj.keys()):
            child_path = f"{_path}.{k}" if _path else k
            out[k] = _canonicalize(obj[k], set_keys, child_path)
        return out
    if isinstance(obj, (list, tuple)):
        items = [_canonicalize(v, set_keys, _path) for v in obj]
        # A param declared a *set* is order-insensitive → sort for a stable key.
        if _path in set_keys:
            items = sorted(items, key=lambda v: _stable_json(v))
        return items
    return _canonical_number(obj)


def _stable_json(obj: Any) -> str:
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonicalize_params(
    params: dict | None, *, set_keys: Iterable[str] = (), defaults: dict | None = None
) -> str:
    """Canonical JSON string for a params dict (sorted keys, canonical numbers).

    Per the spec, the op MUST fill omitted params from its declared default table *for that
    op_version* BEFORE hashing (don't drop defaults — that couples the key to the live
    default table). Pass that table as ``defaults``; caller-supplied ``params`` override it.
    """
    merged = {**(defaults or {}), **(params or {})}
    canon = _canonicalize(merged, frozenset(set_keys))
    return _stable_json(canon)


def canonical_json(obj: Any) -> bytes:
    """Stable canonical-JSON bytes for any JSON-able object (used by id minting)."""
    return _stable_json(_canonicalize(obj, frozenset())).encode("utf-8")


def memo_key(
    op: str,
    op_version: str,
    input_hashes: Iterable[str],
    params: dict | None = None,
    *,
    env_fingerprint: str = "",
    set_keys: Iterable[str] = (),
    defaults: dict | None = None,
) -> str:
    """Compute the pipeline memo key. ``input_hashes`` order is normalized (sorted).

    The variable-length input-hash list is **length-framed** (count + per-hash length
    prefix), NOT delimiter-joined — a bare ``\\x00`` separator would let ``['x', 'y']`` and
    ``['x\\x00y']`` collide into one key (a stale-cache-serving hazard).
    """
    hasher = blake3()
    hasher.update(op.encode("utf-8"))
    hasher.update(_SEP)
    hasher.update(op_version.encode("utf-8"))
    hasher.update(_SEP)
    hashes = sorted(input_hashes)
    hasher.update(struct.pack("<I", len(hashes)))
    for h in hashes:
        hb = h.encode("utf-8")
        hasher.update(struct.pack("<I", len(hb)))
        hasher.update(hb)
    hasher.update(_SEP)
    hasher.update(canonicalize_params(params, set_keys=set_keys, defaults=defaults).encode("utf-8"))
    hasher.update(_SEP)
    hasher.update(env_fingerprint.encode("utf-8"))
    return "blake3:" + hasher.hexdigest()


def tool_version_fingerprint(*commands: list[str]) -> str:
    """Fingerprint shell-out tool versions (e.g. ``["ffmpeg", "-version"]``).

    Hashes the combined version output of each command. A tool that fails to run
    contributes its command + error marker, so an absent tool still yields a stable
    (distinct) fingerprint rather than crashing the memo key.
    """
    hasher = blake3()
    for cmd in commands:
        hasher.update(("\x00".join(cmd)).encode("utf-8"))
        hasher.update(b"=")
        try:
            out = subprocess.run(
                cmd, capture_output=True, timeout=10, check=False
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:  # absent / unrunnable tool
            out = f"__unavailable__:{exc}".encode("utf-8")
        hasher.update(out)
        hasher.update(_SEP)
    return hasher.hexdigest()
