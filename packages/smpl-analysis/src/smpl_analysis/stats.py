"""Role-corpus statistics (ticket vault-3tuy).

Drains the `feature` frames of a *curated corpus* (a directory / tag / pipe of samples that
share a role — "kick", "pad", "hat", whatever the caller decides) and reduces each feature
key to a robust distribution: ``{median, MAD, p10, p50, p90, n}``. Those become the reference
a scorer compares a candidate against ("is this kick's attack normal for kicks?").

**Generic by construction.** This op bakes in NO role names and NO thresholds — the caller
supplies the corpus and names the role; the numbers fall out of the data. That keeps the
tool a process-free primitive (the layering rule): opinion lives in the skills that call it.

**Log domain.** Perceptual audio quantities are multiplicative, so the stats are computed in
the log domain: a strictly-positive key (ms, Hz, ratios, linear amplitudes) is transformed
with ``log10`` before the median/MAD/percentiles; a key that is already logarithmic or signed
(dB, slopes that can go negative) is used as-is (``domain: "linear"``). Each key records which
domain it was reduced in, so a consumer knows whether to exponentiate.

**MUST carry n.** Every key's stats and the corpus summary carry ``n`` — a median over 3
samples is not a median over 300, and a scorer needs to know.

**Caching.** The result is content-addressed (the JSON blob lands in the CAS) and memo-keyed
on ``(corpus content set, op_version)`` — rebuilding over an unchanged corpus is a cache hit.
The reduced stats are ALSO written to ``~/.smpl/stats/<role>.json`` (override the directory
with ``SMPL_STATS_DIR``) so other tools can read a role's reference without replaying a pipe.

Pure-ish: the reduction is pure; the CAS write, memo index, and role-file write are the only
side effects, all confined to :func:`build_stats`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

OP = "stats"
OP_VERSION = "stats@1"

# Percentiles reported per key (p50 == median is carried explicitly too, per the spec list).
_PCTL = (10, 50, 90)


def stats_dir() -> Path:
    """Resolve the role-stats directory (reads ``SMPL_STATS_DIR`` each call so tests override)."""
    return Path(os.environ.get("SMPL_STATS_DIR", "~/.smpl/stats")).expanduser()


def _scalar_of(value: Any) -> Optional[float]:
    """Extract a representative numeric scalar from a feature value, or None if not numeric.

    Handles the registry's shapes: a bare number; a ``{mean, stdev, …}`` stat object (→ mean);
    a ``{median, …}`` object (→ median). Booleans and non-numeric values are skipped (a scorer
    reduces distributions of *numbers*, not flags/strings).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        import math

        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        for pick in ("mean", "median", "p50"):
            if pick in value:
                return _scalar_of(value[pick])
    return None


def collect_key_values(frames: list[dict]) -> dict[str, list[float]]:
    """Group every finite scalar in the corpus by feature key (one value per feature frame)."""
    acc: dict[str, list[float]] = {}
    for f in frames:
        if f.get("kind") != "feature":
            continue
        data = f.get("data")
        if not isinstance(data, dict):
            continue
        for key, val in data.items():
            scalar = _scalar_of(val)
            if scalar is None:
                continue
            acc.setdefault(key, []).append(scalar)
    return acc


def _reduce(values: list[float]) -> dict:
    """Reduce a list of scalars to ``{median, mad, p10, p50, p90, n, domain}`` in log domain.

    A strictly-positive key is reduced in ``log10`` space; anything with a non-positive value
    (dB, signed slopes) is reduced linearly. MAD is the median absolute deviation about the
    median, in the same domain.
    """
    import numpy as np

    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"median": None, "mad": None, "p10": None, "p50": None, "p90": None, "n": 0,
                "domain": "linear"}

    if np.all(arr > 0.0):
        domain = "log10"
        x = np.log10(arr)
    else:
        domain = "linear"
        x = arr

    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    p10, p50, p90 = (float(np.percentile(x, p)) for p in _PCTL)
    return {
        "median": round(median, 6),
        "mad": round(mad, 6),
        "p10": round(p10, 6),
        "p50": round(p50, 6),
        "p90": round(p90, 6),
        "n": n,
        "domain": domain,
    }


def reduce_corpus(frames: list[dict]) -> dict[str, dict]:
    """Reduce a corpus of feature frames to ``{feature_key: {median, mad, p10, p50, p90, n, domain}}``."""
    return {key: _reduce(vals) for key, vals in sorted(collect_key_values(frames).items())}


def corpus_content_set(frames: list[dict]) -> list[str]:
    """The corpus's content identity: the sorted set of source audio CAS hashes.

    Falls back to the set of feature-frame ids when no audio frames are present in the stream
    (e.g. a pre-reduced feature pipe), so the memo key is always well-defined.
    """
    hashes = sorted({f["hash"] for f in frames
                     if f.get("kind") == "audio" and f.get("hash")})
    if hashes:
        return hashes
    return sorted({f["id"] for f in frames
                   if f.get("kind") == "feature" and f.get("id")})


def _corpus_n(frames: list[dict], per_key: dict[str, dict]) -> int:
    """Corpus size: the number of source audio frames, or the max per-key ``n`` as a fallback."""
    n_audio = sum(1 for f in frames if f.get("kind") == "audio")
    if n_audio:
        return n_audio
    return max((v.get("n", 0) for v in per_key.values()), default=0)


def _memo_index_path() -> Path:
    return stats_dir() / "index.json"


def _read_memo_index() -> dict:
    p = _memo_index_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_memo_index(index: dict) -> None:
    p = _memo_index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, sort_keys=True, indent=2))


def build_stats(frames: list[dict], *, role: str) -> list[dict]:
    """Build role-corpus stats from a stream of feature frames; return derived frame(s).

    Reduces the corpus, memo-keys on ``(corpus content set, op_version)``, content-addresses
    the JSON result in the CAS, records the memo→CAS mapping, and writes
    ``<stats_dir>/<role>.json``. Returns a one-element list with a `feature` frame
    (role ``stats:<role>``) whose ``data`` is the per-key stats and whose ``params`` carry the
    role, corpus n, memo key, CAS hash, and whether the build was a cache hit.
    """
    from smplstream import cas, frames as F, memo

    per_key = reduce_corpus(frames)
    content_set = corpus_content_set(frames)
    n = _corpus_n(frames, per_key)

    mkey = memo.memo_key(OP, OP_VERSION, content_set, params={})

    index = _read_memo_index()
    cached_hash = index.get(mkey)
    cache_hit = bool(cached_hash and cas.exists(cached_hash))

    result = {
        "role": role,
        "op_version": OP_VERSION,
        "memo_key": mkey,
        "n": n,
        "corpus_size": len(content_set),
        "stats": per_key,
    }

    if cache_hit:
        try:
            cached = json.loads(cas.get_path(cached_hash).read_text())
            # Reuse the reduced numbers (corpus identity is unchanged); re-label with this role.
            per_key = cached.get("stats", per_key)
            n = cached.get("n", n)
            result["stats"] = per_key
            result["n"] = n
            cas_hash = cached_hash
        except (OSError, json.JSONDecodeError):
            cache_hit = False

    if not cache_hit:
        blob = json.dumps(result, sort_keys=True).encode("utf-8")
        cas_hash = cas.put_blob(blob, "application/json")
        index[mkey] = cas_hash
        _write_memo_index(index)

    # Always (re)write the role file so a role's reference is current on disk.
    role_path = stats_dir() / f"{role}.json"
    role_path.parent.mkdir(parents=True, exist_ok=True)
    role_path.write_text(json.dumps({**result, "cas_hash": cas_hash}, sort_keys=True, indent=2))

    params = {"role": role, "n": n, "corpus_size": len(content_set),
              "memo_key": mkey, "cas_hash": cas_hash, "cache_hit": cache_hit}
    frame = F.feature_frame(
        per_key, role=f"stats:{role}", op=OP, op_version=OP_VERSION, params=params
    )
    return [frame]
