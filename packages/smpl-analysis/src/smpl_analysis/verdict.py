"""The gen-QC judge op (ticket vault-1apl) — emits `verdict` frames.

The native output of the generation quality-control gate. For each candidate `audio` frame in
a stream it reads that candidate's `feature` frames, scores them against a per-role collection
profile (``smpl stats build``), and emits ONE `verdict` frame conforming to
``music/smplstream/verdict-frames.md``: discretized per-axis verdicts (5-token enum) plus a
routing rollup (keep | listen | cut | alter). The deterministic tier does the arithmetic; the
model reads tokens, not z-scores.

**Reuses the ONE scoring core** (:mod:`smpl_analysis.triage`) — ``score_candidate`` /
``robust_z`` / ``axis_band`` / ``candidate_features``. The whole gen-QC + curation + contact-
sheet stack rides one z/band computation; this module does not fork it. The band constants are
triage's (``Z_IN_BAND`` = 2.0, ``Z_FAR`` = 3.5).

**Routing** (off the worst axis ``max_abs``):

    max_abs < 2.0            → keep    (in tolerance, auto-accept)
    2.0 <= max_abs < 3.5     → alter   (the repairable middle band → the repair loop)
    max_abs >= 3.5           → cut     (a hard fail)

**Confidence** (enum, never a float): ``low`` when the corpus is thin (min overlapping axis
n < 12 → *insufficient_corpus*) or no keys overlap; ``high`` only when the call is unambiguous
(clearly in-band or clearly far, off the boundary by ``CONF_MARGIN``) AND n is adequate; else
``med``.

**The three-band lever**: a low-confidence keep/cut is *downgraded to* ``listen`` — the human
ear is spent only where the numbers are weak. A thin corpus (n < 12) therefore never
auto-keeps/cuts.

**Layering (verdict-frames.md, Nick 2026-07-02).** This tool emits z / per-axis verdict /
rollup ONLY. It deliberately LEAVES OUT the ``try`` fix-table resolution: which EQ/gain/limit
move repairs a given failing axis lives in ``music/sample-generation/fix-table.md`` and the
/sample-gen skill, not in the tool. The gate says *what* is wrong and *how far*; the skill
resolves the fix.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from smplstream import SCHEMA

from . import triage as T

OP = "judge"
OP_VERSION = "judge@0.1"

# Corpus-size floor for a trustworthy relative judgment (verdict-frames.md n-gating table).
MIN_CORPUS_N = 12

# How far off a band boundary a call must be to count as "unambiguous" (→ high confidence).
# Clearly in-band: max_abs < Z_IN_BAND - CONF_MARGIN; clearly far: max_abs >= Z_FAR + CONF_MARGIN.
CONF_MARGIN = 0.5

# Unit suffix → display unit, so an axis carries `unit` when the key names one (spec: optional).
_UNIT_SUFFIX = {
    "_lufs": "LUFS", "_dbtp": "dBTP", "_dbfs": "dBFS", "_db": "dB", "_hz": "Hz", "_ms": "ms",
}


# ---------------------------------------------------------------------------------------------
# Loading (profile + optional brief). Scoring itself is triage's; this is just I/O.
# ---------------------------------------------------------------------------------------------


def load_profile(path: str) -> tuple[dict, dict]:
    """Load a role stats JSON; return ``(per_key_stats, meta)``.

    ``meta`` carries ``{role, n, corpus_size, cas_hash, op_version}`` from the artifact
    top-level (for the ``stats`` label, corpus-n confidence gate, and lineage). Accepts the full
    artifact ``smpl stats build`` writes or a bare per-key dict.
    """
    doc = json.loads(Path(path).expanduser().read_text())
    if isinstance(doc, dict) and isinstance(doc.get("stats"), dict):
        per_key = doc["stats"]
        meta = {k: doc.get(k) for k in ("role", "n", "corpus_size", "cas_hash", "op_version")}
    else:
        per_key = doc if isinstance(doc, dict) else {}
        meta = {"role": None, "n": None, "corpus_size": None, "cas_hash": None, "op_version": None}
    return per_key, meta


def load_target(path: str) -> dict:
    """Load an absolute-tolerance brief; return ``{key: (lo, hi)}``.

    A brief is the caller's absolute target band (bootstrap/fallback reference frame, separate
    from the collection profile). Accepts ``{key: [lo, hi]}`` or ``{key: {"min": lo, "max": hi}}``.
    """
    doc = json.loads(Path(path).expanduser().read_text())
    out: dict[str, tuple[float, float]] = {}
    if not isinstance(doc, dict):
        return out
    for key, band in doc.items():
        if isinstance(band, (list, tuple)) and len(band) == 2:
            out[key] = (float(band[0]), float(band[1]))
        elif isinstance(band, dict) and "min" in band and "max" in band:
            out[key] = (float(band["min"]), float(band["max"]))
    return out


def _unit_of(key: str) -> Optional[str]:
    for suffix, unit in _UNIT_SUFFIX.items():
        if key.endswith(suffix):
            return unit
    return None


def stats_label(meta: dict) -> str:
    role = meta.get("role") or "?"
    n = meta.get("n")
    return f"collection-stats:{role}" + (f" (n={n})" if n is not None else "")


def stats_lineage(meta: dict) -> str:
    """A lineage token for the reference profile: its CAS hash if known, else a role label."""
    h = meta.get("cas_hash")
    if isinstance(h, str) and h:
        return h
    return f"collection-stats:{meta.get('role') or '?'}"


# ---------------------------------------------------------------------------------------------
# The judge (pure): score → axes[] + rollup. Reuses triage.score_candidate wholesale.
# ---------------------------------------------------------------------------------------------


def _rollup_score(distance: Optional[float]) -> Optional[int]:
    """Optional 0-100 rollup score, a deterministic f(distance). 100 at distance 0; monotone."""
    if distance is None or not math.isfinite(distance):
        return None
    return int(round(100.0 / (1.0 + distance)))


def judge(feat: dict, profile: dict, meta: dict, *, target: Optional[dict] = None) -> dict:
    """Judge one candidate's flat feature dict against a role profile; return the verdict ``data``.

    Pure. Delegates all z/band arithmetic to :func:`triage.score_candidate` (one core), then
    discretizes into the axes[] + rollup shape of verdict-frames.md.
    """
    score = T.score_candidate(feat, profile)
    per_key = score["per_key"]
    max_abs = score["max_abs"]
    n_keys = score["n_keys"]

    # --- corpus adequacy: min n across the axes that actually overlapped. ---
    overlap_ns = [profile[k].get("n") for k in per_key if isinstance(profile.get(k), dict)]
    overlap_ns = [int(n) for n in overlap_ns if n is not None]
    min_n = min(overlap_ns) if overlap_ns else 0
    thin = (n_keys == 0) or (min_n < MIN_CORPUS_N)

    # --- base decision from the worst axis (three-band routing). ---
    if max_abs < T.Z_IN_BAND:
        decision = "keep"
    elif max_abs < T.Z_FAR:
        decision = "alter"
    else:
        decision = "cut"

    # --- confidence (enum) + reason. ---
    reason: Optional[str] = None
    if n_keys == 0:
        confidence, reason = "low", "no_overlap"
    elif min_n < MIN_CORPUS_N:
        confidence, reason = "low", "insufficient_corpus"
    else:
        unambiguous = (max_abs < T.Z_IN_BAND - CONF_MARGIN) or (max_abs >= T.Z_FAR + CONF_MARGIN)
        confidence = "high" if unambiguous else "med"

    # --- the three-band lever: a low-confidence keep/cut spends a human listen instead. ---
    if confidence == "low" and decision in ("keep", "cut"):
        decision = "listen"

    distance = None if math.isinf(score["agg"]) else round(score["agg"], 4)
    dominant_axis = score["worst"][0][0] if score["worst"] else None

    rollup = {
        "decision": decision,
        "confidence": confidence,
        "dominant_axis": dominant_axis,
        "distance": distance,
        "score": _rollup_score(distance),
    }
    if reason is not None:
        rollup["reason"] = reason

    # --- axes[]: one entry per overlapping key, worst-first; in-band axes included + marked. ---
    axes = []
    for key, z, band in score["worst"]:
        stat = profile.get(key) or {}
        axis = {
            "key": key,
            "measured": round(float(feat[key]), 6),
            "ref": {"median": stat.get("median"), "mad": stat.get("mad"), "n": stat.get("n")},
            "z": round(z, 4),
            "verdict": band,
        }
        unit = _unit_of(key)
        if unit is not None:
            axis["unit"] = unit
        if target is not None and key in target:
            lo, hi = target[key]
            axis["in_tolerance"] = bool(lo <= float(feat[key]) <= hi)
        axes.append(axis)

    data = {"stats": stats_label(meta), "rollup": rollup, "axes": axes}
    if target is not None:
        data["target"] = "brief:absolute-tolerances"
    return data


# ---------------------------------------------------------------------------------------------
# Frame builders (impure only in that they mint ids / gather lineage from the stream).
# ---------------------------------------------------------------------------------------------


def judge_audio_frame(
    frames: list[dict],
    audio: dict,
    profile: dict,
    meta: dict,
    *,
    index: int = 0,
    target: Optional[dict] = None,
    schema_version: str = SCHEMA,
) -> dict:
    """Build the `verdict` frame for one candidate `audio` frame in the stream."""
    from smplstream import frames as F

    feat = T.candidate_features(frames, audio["id"])
    feature_ids = [
        f["id"] for f in frames
        if f.get("kind") == "feature" and f.get("of") == audio["id"] and f.get("id")
    ]
    label = T.candidate_label(audio, index)
    data = judge(feat, profile, meta, target=target)
    lineage = feature_ids + [stats_lineage(meta)]
    return F.verdict_frame(
        data, of=audio["id"], lineage=lineage, schema_version=schema_version,
        role=f"candidate:{label}", op=OP, op_version=OP_VERSION,
        params={"stats_role": meta.get("role"), "corpus_n": meta.get("n")},
    )


def verdict_frames(
    frames: list[dict],
    profile: dict,
    meta: dict,
    *,
    role: Optional[str] = None,
    target: Optional[dict] = None,
    schema_version: str = SCHEMA,
) -> list[dict]:
    """Emit one `verdict` frame per candidate `audio` frame in the stream (in stream order).

    ``role`` restricts which audio frames are treated as candidates (default: all audio frames).
    """
    audios = [
        f for f in frames
        if f.get("kind") == "audio" and f.get("id") and (role is None or f.get("role") == role)
    ]
    return [
        judge_audio_frame(frames, af, profile, meta, index=i, target=target,
                          schema_version=schema_version)
        for i, af in enumerate(audios)
    ]
