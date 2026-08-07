"""Verdict backtest / calibration harness (ticket vault-2kyt).

Measures the gen-QC gate (:mod:`smpl_analysis.verdict`, ticket vault-1apl) against a corpus of
**human-labeled** verdicts. The gate is only trustworthy if its ``keep`` calls match the human's
``keep`` calls — a gate you can't trust drives *listens UP*. This module quantifies that trust:

  * a **confusion matrix** (human × gate over ``{keep, listen, cut}``),
  * **precision / recall / f1 per class**,
  * the headline **auto-keep accuracy** — of the samples the gate auto-kept, how many the human
    kept (the ≥ 0.85 calibration target from vault-1apl),
  * **auto-cut accuracy** (the mirror), and **overall accuracy**,
  * **per-axis contribution** — which feature axis drove each verdict and how those calls landed
    against the human labels (surfaces the axes that carry / lose trust),
  * a **threshold-sensitivity** readout — how auto-keep would move if ``Z_IN_BAND`` shifted ±0.5
    and how auto-cut would move if ``Z_FAR`` shifted ±0.5 (the tuning levers, measured, not
    guessed).

**Layering (the harness is a script + doc thresholds; nothing baked into ``smpl``).** This module
reuses the ONE scoring core — it calls :func:`smpl_analysis.verdict.judge` (which rides
:mod:`smpl_analysis.triage`) for the *real* gate decision, and re-derives decisions parametrically
only for the sensitivity sweep. The band constants it perturbs are triage's (``Z_IN_BAND`` = 2.0,
``Z_FAR`` = 3.5); the corpus, roles, and the 0.85 target are the caller's (doc/skill), not baked
in here.

**Alter folds to listen.** The gate emits four tokens (keep | listen | cut | alter); the corpus
uses three (keep | listen | cut). ``alter`` (the repairable middle band) and ``listen`` (the
low-confidence middle) are both "not auto-decided — human/repair in the loop", so for the
three-class confusion matrix ``alter`` folds into ``listen``. The raw four-token decision is kept
per entry so nothing is lost.

**Schema staleness.** Each corpus entry may carry ``feature_schema`` (the ``smplstream`` SCHEMA
in effect when it was labeled). Entries whose stamp mismatches the current schema are flagged in
the report (``stale_schema``) — the corpus rots the first time feature definitions change, and
this is the tripwire.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Union

from smplstream import SCHEMA

from . import triage as T
from . import verdict as V

# The three corpus classes (the confusion-matrix axes). Gate `alter` folds into `listen`.
CLASSES = ("keep", "listen", "cut")

# The vault-1apl calibration gate: of the gate's auto-keeps, ≥ 85% must be human keeps.
TARGET_AUTO_KEEP = 0.85

# Threshold-sensitivity sweep offsets around the triage band constants (±0.5, the tuning lever).
SENS_OFFSETS = (-0.5, 0.0, 0.5)

# A profile resolver maps a role name to ``(per_key_profile, meta)`` (or None when unavailable).
ProfileResolver = Union[dict, Callable[[str], Optional[tuple]]]


# ---------------------------------------------------------------------------------------------
# Corpus loading.
# ---------------------------------------------------------------------------------------------


def load_corpus(path: str) -> list[dict]:
    """Load a verdict corpus from JSONL (one entry per line) or a JSON array.

    Each entry: ``{id, role, human_verdict, features?|sample_path?, note?, feature_schema?}``.
    Blank lines and ``#`` comment lines are skipped (JSONL). Raises on malformed JSON.
    """
    text = Path(path).expanduser().read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        doc = json.loads(text)
        return list(doc) if isinstance(doc, list) else []
    entries: list[dict] = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"corpus line {i}: {exc}") from exc
    return entries


def features_for_entry(entry: dict) -> dict:
    """Resolve an entry's flat feature dict — precomputed ``features``, else from ``sample_path``.

    A precomputed ``features`` map makes the corpus runnable *without audio* (fixtures, CI). A
    ``sample_path`` is resolved through the same feature path the gate uses at judgment time
    (``describe`` light tier → :func:`triage.candidate_features`), so a real corpus stays honest.
    """
    feats = entry.get("features")
    if isinstance(feats, dict) and feats:
        return {k: float(v) for k, v in feats.items()}
    sample_path = entry.get("sample_path")
    if not sample_path:
        raise ValueError(f"entry {entry.get('id')!r}: neither `features` nor `sample_path`")

    from smplstream import cas, frames as F

    from . import describe as D

    h = cas.put_audio_file(str(Path(sample_path).expanduser()))
    meta = cas.read_meta(h) or {}
    af = F.audio_frame(
        h, sr=meta.get("sr", 0), ch=meta.get("ch", 1), dur=meta.get("dur", 0.0), role="source"
    )
    derived = D.describe_audio_frame(af, want_image=False)
    return T.candidate_features([af, *derived], af["id"])


# ---------------------------------------------------------------------------------------------
# Decision logic (parametric mirror of verdict.judge, for the sensitivity sweep).
# ---------------------------------------------------------------------------------------------


def fold3(decision: str) -> str:
    """Fold the gate's four-token decision into the three corpus classes (``alter`` → ``listen``)."""
    return "listen" if decision == "alter" else decision


def decide(
    max_abs: float,
    min_n: int,
    n_keys: int,
    *,
    z_in_band: float,
    z_far: float,
    min_corpus_n: int = V.MIN_CORPUS_N,
    conf_margin: float = V.CONF_MARGIN,
) -> tuple[str, str]:
    """Reproduce :func:`verdict.judge`'s routing + confidence for arbitrary band thresholds.

    Returns ``(decision, confidence)`` where ``decision`` is the raw four-token routing (keep |
    alter | cut | listen). This is an exact mirror of the judge's decision logic (a test pins it
    to the real gate at the baseline thresholds) so the sensitivity sweep can perturb the bands
    without mutating module globals.
    """
    if max_abs < z_in_band:
        decision = "keep"
    elif max_abs < z_far:
        decision = "alter"
    else:
        decision = "cut"

    if n_keys == 0:
        confidence = "low"
    elif min_n < min_corpus_n:
        confidence = "low"
    else:
        unambiguous = (max_abs < z_in_band - conf_margin) or (max_abs >= z_far + conf_margin)
        confidence = "high" if unambiguous else "med"

    if confidence == "low" and decision in ("keep", "cut"):
        decision = "listen"
    return decision, confidence


def _overlap_min_n(feat: dict, profile: dict, per_key: dict) -> int:
    """Minimum corpus n across the profiled axes the candidate actually overlapped (judge's rule)."""
    ns = [
        int(profile[k]["n"])
        for k in per_key
        if isinstance(profile.get(k), dict) and profile[k].get("n") is not None
    ]
    return min(ns) if ns else 0


# ---------------------------------------------------------------------------------------------
# Scoring a corpus.
# ---------------------------------------------------------------------------------------------


def _resolve_profile(resolver: ProfileResolver, role: Optional[str]) -> Optional[tuple]:
    if callable(resolver):
        return resolver(role)
    if isinstance(resolver, dict):
        # A single (profile, meta) supplied for EVERY role (the apply-to-all shape).
        if "profile" in resolver and "meta" in resolver:
            return (resolver["profile"], resolver["meta"])
        # Otherwise a role-keyed dict: match by role exactly (no cross-role fallback).
        if role is not None and role in resolver:
            return resolver[role]
    return None


def score_entries(
    corpus: list[dict],
    profiles: ProfileResolver,
    *,
    role: Optional[str] = None,
    schema: str = SCHEMA,
) -> tuple[list[dict], list[dict]]:
    """Run the gate over every corpus entry; return ``(scored, skipped)``.

    ``scored`` entries carry the human + gate verdicts, confidence, dominant axis, and the raw
    scoring numbers (``max_abs`` / ``min_n`` / ``n_keys``) the sensitivity sweep needs. ``skipped``
    entries carry a ``reason`` (no profile for the role, no features, no human label).
    """
    scored: list[dict] = []
    skipped: list[dict] = []

    for entry in corpus:
        erole = entry.get("role")
        if role is not None and erole != role:
            continue

        human = entry.get("human_verdict")
        if human is None:
            skipped.append({"id": entry.get("id"), "role": erole, "reason": "no_human_verdict"})
            continue

        resolved = _resolve_profile(profiles, erole)
        if resolved is None:
            skipped.append({"id": entry.get("id"), "role": erole, "reason": "no_profile"})
            continue
        profile, meta = resolved

        try:
            feat = features_for_entry(entry)
        except Exception as exc:
            skipped.append({"id": entry.get("id"), "role": erole, "reason": f"features: {exc}"})
            continue

        sc = T.score_candidate(feat, profile)
        n_keys = sc["n_keys"]
        max_abs = sc["max_abs"]
        min_n = _overlap_min_n(feat, profile, sc["per_key"])

        data = V.judge(feat, profile, meta)
        rollup = data["rollup"]
        gate_raw = rollup["decision"]

        entry_schema = entry.get("feature_schema")
        stale = bool(entry_schema) and entry_schema != schema

        scored.append(
            {
                "id": entry.get("id"),
                "role": erole,
                "human": fold3(str(human)),
                "human_raw": human,
                "gate": fold3(gate_raw),
                "gate_raw": gate_raw,
                "confidence": rollup["confidence"],
                "dominant_axis": rollup.get("dominant_axis"),
                "max_abs": None if _isinf(max_abs) else round(max_abs, 4),
                "_max_abs": max_abs,  # kept unrounded (may be inf) for the sensitivity sweep
                "min_n": min_n,
                "n_keys": n_keys,
                "agree": fold3(str(human)) == fold3(gate_raw),
                "note": entry.get("note"),
                "stale_schema": stale,
                "feature_schema": entry_schema,
            }
        )
    return scored, skipped


def _isinf(x) -> bool:
    import math

    return isinstance(x, float) and math.isinf(x)


# ---------------------------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------------------------


def confusion_matrix(scored: list[dict]) -> dict:
    """``cm[human][gate]`` counts over the three classes (rows = human, cols = gate)."""
    cm = {h: {g: 0 for g in CLASSES} for h in CLASSES}
    for e in scored:
        h, g = e["human"], e["gate"]
        if h in cm and g in cm[h]:
            cm[h][g] += 1
    return cm


def _safe_ratio(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def per_class_metrics(cm: dict) -> dict:
    """Precision / recall / f1 + support per class from a confusion matrix (rows = human)."""
    out: dict[str, dict] = {}
    for c in CLASSES:
        tp = cm[c][c]
        gate_total = sum(cm[h][c] for h in CLASSES)  # everything the gate called c (column)
        human_total = sum(cm[c][g] for g in CLASSES)  # everything the human called c (row)
        precision = _safe_ratio(tp, gate_total)
        recall = _safe_ratio(tp, human_total)
        if precision and recall and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        elif precision is not None and recall is not None:
            f1 = 0.0
        else:
            f1 = None
        out[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_human": human_total,
            "support_gate": gate_total,
        }
    return out


def per_axis_contribution(scored: list[dict]) -> dict:
    """Aggregate per dominant axis: how often it drove a verdict and how those calls landed.

    ``{axis: {dominant_count, agree, human: {keep,listen,cut}, gate: {keep,listen,cut}}}`` — the
    axes that carry (or lose) trust. ``agree`` is the fraction of that axis's entries where gate
    matched human.
    """
    acc: dict[str, dict] = {}
    for e in scored:
        ax = e["dominant_axis"]
        if ax is None:
            ax = "(none)"
        slot = acc.setdefault(
            ax,
            {
                "dominant_count": 0,
                "n_agree": 0,
                "human": {c: 0 for c in CLASSES},
                "gate": {c: 0 for c in CLASSES},
            },
        )
        slot["dominant_count"] += 1
        slot["n_agree"] += 1 if e["agree"] else 0
        if e["human"] in slot["human"]:
            slot["human"][e["human"]] += 1
        if e["gate"] in slot["gate"]:
            slot["gate"][e["gate"]] += 1
    for slot in acc.values():
        slot["agree"] = _safe_ratio(slot["n_agree"], slot["dominant_count"])
    return acc


def threshold_sensitivity(scored: list[dict]) -> dict:
    """Auto-keep vs ``Z_IN_BAND`` and auto-cut vs ``Z_FAR``, each swept ±0.5 (the tuning levers).

    Auto-keep is (near-)insensitive to ``Z_FAR`` and auto-cut to ``Z_IN_BAND`` by construction —
    ``Z_IN_BAND`` sets the keep boundary, ``Z_FAR`` the cut boundary — so the sweep pairs each
    metric with the lever that moves it, and reports how each would shift the headline number.
    """

    def _auto_keep_at(z_in_band: float, z_far: float) -> tuple[Optional[float], int]:
        keeps = [
            e for e in scored
            if decide(e["_max_abs"], e["min_n"], e["n_keys"],
                      z_in_band=z_in_band, z_far=z_far)[0] == "keep"
        ]
        hits = sum(1 for e in keeps if e["human"] == "keep")
        return _safe_ratio(hits, len(keeps)), len(keeps)

    def _auto_cut_at(z_in_band: float, z_far: float) -> tuple[Optional[float], int]:
        cuts = [
            e for e in scored
            if decide(e["_max_abs"], e["min_n"], e["n_keys"],
                      z_in_band=z_in_band, z_far=z_far)[0] == "cut"
        ]
        hits = sum(1 for e in cuts if e["human"] == "cut")
        return _safe_ratio(hits, len(cuts)), len(cuts)

    z_in_rows = []
    for off in SENS_OFFSETS:
        z = round(T.Z_IN_BAND + off, 4)
        acc, n = _auto_keep_at(z, T.Z_FAR)
        z_in_rows.append({"value": z, "offset": off, "auto_keep_accuracy": acc,
                          "n_gate_keep": n, "baseline": off == 0.0})

    z_far_rows = []
    for off in SENS_OFFSETS:
        z = round(T.Z_FAR + off, 4)
        acc, n = _auto_cut_at(T.Z_IN_BAND, z)
        z_far_rows.append({"value": z, "offset": off, "auto_cut_accuracy": acc,
                           "n_gate_cut": n, "baseline": off == 0.0})

    return {"z_in_band": z_in_rows, "z_far": z_far_rows}


# ---------------------------------------------------------------------------------------------
# The report.
# ---------------------------------------------------------------------------------------------


def run_backtest(
    corpus: list[dict],
    profiles: ProfileResolver,
    *,
    role: Optional[str] = None,
    target_auto_keep: float = TARGET_AUTO_KEEP,
    corpus_path: Optional[str] = None,
    schema: str = SCHEMA,
) -> dict:
    """Backtest the gate over a corpus; return the full JSON-able calibration report."""
    scored, skipped = score_entries(corpus, profiles, role=role, schema=schema)

    cm = confusion_matrix(scored)
    per_class = per_class_metrics(cm)
    total = len(scored)
    correct = sum(cm[c][c] for c in CLASSES)

    auto_keep = per_class["keep"]["precision"]
    auto_cut = per_class["cut"]["precision"]
    roles = sorted({e["role"] for e in scored if e["role"] is not None})
    stale = [e["id"] for e in scored if e["stale_schema"]]

    entries = [
        {k: v for k, v in e.items() if not k.startswith("_")}
        for e in scored
    ]

    return {
        "corpus": corpus_path,
        "schema": schema,
        "n_entries": len(corpus),
        "n_scored": total,
        "n_skipped": len(skipped),
        "roles": roles,
        "classes": list(CLASSES),
        "confusion": cm,
        "per_class": per_class,
        "auto_keep_accuracy": auto_keep,
        "auto_cut_accuracy": auto_cut,
        "overall_accuracy": _safe_ratio(correct, total),
        "target_auto_keep": target_auto_keep,
        "meets_target": (auto_keep is not None and auto_keep >= target_auto_keep),
        "per_axis": per_axis_contribution(scored),
        "threshold_sensitivity": threshold_sensitivity(scored),
        "stale_schema_ids": stale,
        "skipped": skipped,
        "entries": entries,
    }


# ---------------------------------------------------------------------------------------------
# Human-readable rendering.
# ---------------------------------------------------------------------------------------------


def _pct(x: Optional[float]) -> str:
    return "  —  " if x is None else f"{x:5.2f}"


def format_report(report: dict) -> str:
    """A compact, legible summary of the calibration report (the scalar strip, not the JSON)."""
    L: list[str] = []
    corpus = report.get("corpus") or "(in-memory)"
    L.append(f"verdict backtest — corpus: {corpus}")
    L.append(
        f"  entries: {report['n_scored']} scored ({report['n_skipped']} skipped)"
        f"   roles: {', '.join(report['roles']) or '—'}"
        f"   schema: {report['schema']}"
    )
    if report.get("stale_schema_ids"):
        L.append(f"  ! stale feature_schema on: {', '.join(report['stale_schema_ids'])}")
    L.append("")

    cm = report["confusion"]
    L.append("confusion matrix (rows = human, cols = gate; alter folded → listen)")
    L.append("               gate:keep  gate:listen  gate:cut  │ human tot")
    for h in CLASSES:
        row = cm[h]
        tot = sum(row.values())
        L.append(
            f"  human:{h:<7}{row['keep']:>6}{row['listen']:>12}{row['cut']:>10}   │{tot:>6}"
        )
    gate_tot = {g: sum(cm[h][g] for h in CLASSES) for g in CLASSES}
    L.append("  " + "─" * 52)
    L.append(
        f"  gate tot    {gate_tot['keep']:>6}{gate_tot['listen']:>12}{gate_tot['cut']:>10}"
        f"   │{report['n_scored']:>6}"
    )
    L.append("")

    pc = report["per_class"]
    L.append("per-class        precision  recall    f1     support(H/G)")
    for c in CLASSES:
        m = pc[c]
        L.append(
            f"  {c:<12}{_pct(m['precision'])}    {_pct(m['recall'])}  {_pct(m['f1'])}"
            f"      {m['support_human']} / {m['support_gate']}"
        )
    L.append("")

    L.append("headline")
    ak = report["auto_keep_accuracy"]
    flag = "✗ BELOW TARGET" if not report["meets_target"] else "✓ meets target"
    L.append(
        f"  auto-keep accuracy  {_pct(ak)}   (target ≥ {report['target_auto_keep']:.2f})   {flag}"
    )
    L.append(f"  auto-cut  accuracy  {_pct(report['auto_cut_accuracy'])}")
    L.append(f"  overall   accuracy  {_pct(report['overall_accuracy'])}")
    L.append("")

    sens = report["threshold_sensitivity"]
    L.append("threshold sensitivity (the tuning levers, measured ±0.5)")
    for r in sens["z_in_band"]:
        tag = "  [baseline]" if r["baseline"] else ""
        L.append(
            f"  Z_IN_BAND {r['value']:>4} → auto-keep {_pct(r['auto_keep_accuracy'])}"
            f"  (n_keep={r['n_gate_keep']}){tag}"
        )
    for r in sens["z_far"]:
        tag = "  [baseline]" if r["baseline"] else ""
        L.append(
            f"  Z_FAR     {r['value']:>4} → auto-cut  {_pct(r['auto_cut_accuracy'])}"
            f"  (n_cut={r['n_gate_cut']}){tag}"
        )
    L.append("")

    axes = report["per_axis"]
    if axes:
        L.append("dominant-axis contribution (which axis drove the verdict)")
        for ax, slot in sorted(axes.items(), key=lambda kv: -kv[1]["dominant_count"]):
            hb = slot["human"]
            L.append(
                f"  {ax:<28} drove {slot['dominant_count']:>2}"
                f"  → human keep:{hb['keep']} listen:{hb['listen']} cut:{hb['cut']}"
                f"   (agree {_pct(slot['agree'])})"
            )

    return "\n".join(L)
