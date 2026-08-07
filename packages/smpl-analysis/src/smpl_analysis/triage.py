"""Batch-triage visualization layer (ticket vault-22oy) — built for LLM legibility.

Two deliverables, one scoring core:

  1. **Contact sheet** (:func:`render_contact_sheet`) — ONE composite PNG per candidate
     batch. Tiles are sorted best→worst by aggregate robust-z distance from a role profile
     (this sort is deterministic work the model must NOT redo — it is recorded in the image
     frame's ``params.order`` and echoed as a text index). Each tile burns in a verdict band
     (PASS/ALTER/REJECT from |z|) + the 1–2 worst axes as TEXT, because a model reads
     text-in-image far more reliably than it estimates dB off a colormap — the image's job is
     gestalt + anomaly-spotting, not measurement. All tiles share one calibrated dB
     scale / colormap / freq axis (reference gridlines at 60/120/500/2k/8k Hz) and carry a
     border colored by verdict band.

  2. **Spectrum-vs-profile overlay** (:func:`render_profile_overlay`) — the candidate's
     smoothed 1/6-octave spectrum drawn over the role-profile median with a shaded p10–p90
     band, the six band-deltas printed as text.

The **robust-z** is domain-aware, matching what ``smpl stats build`` recorded per key:
a ``log10``-domain key is compared in log space (``z = (log10(x) − median) / MAD``); a
``linear``-domain key (dB / signed) is compared as-is. MAD is the RAW median-absolute-
deviation the stats op stored (no 1.4826 scaling) — the definition the spec pins.

Generic by construction: no roles or thresholds are baked in — the caller supplies a stats
file via ``--stats`` and the numbers fall out of it (the layering rule).

Heavy imports (matplotlib/librosa/numpy) stay inside functions; matplotlib forces the
non-interactive **Agg** backend before pyplot is imported (headless, deterministic).
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Optional

from .stats import _scalar_of  # reuse the exact scalar-extraction the corpus reducer used
from .width import BANDS

OP_SHEET = "contact-sheet"
OP_SHEET_VERSION = "contact-sheet@1"
OP_OVERLAY = "profile-overlay"
OP_OVERLAY_VERSION = "profile-overlay@1"

# Verdict thresholds on |z| (spec): in-band < 2, below/above 2–3.5, far_* ≥ 3.5.
Z_IN_BAND = 2.0
Z_FAR = 3.5
Z_CAP = 12.0          # cap |z| so a zero-variance corpus can't produce inf (kept deterministic)
MAD_FLOOR = 1e-9      # floor MAD so a zero-spread key doesn't divide by zero

# Verdict palette (border + text). Green pass, amber alter, red reject.
_COLOR = {"PASS": "#2e7d32", "ALTER": "#f9a825", "REJECT": "#c62828"}

# STFT/mel defaults for the tiles (match spectrogram.py so the look is consistent).
_N_FFT = 2048
_HOP = 512
_N_MELS = 128
_TOP_DB = 80.0

# Reference gridlines (Hz) drawn on every tile / overlay.
_GRIDLINES = (60.0, 120.0, 500.0, 2000.0, 8000.0)


# ---------------------------------------------------------------------------------------------
# Stats-file loading.
# ---------------------------------------------------------------------------------------------


def load_stats_file(path: str) -> dict:
    """Load a role stats JSON and return its per-key ``{key: {median, mad, ...}}`` map.

    Accepts either the full artifact ``smpl stats build`` writes (``{role, n, stats: {...}}``)
    or a bare per-key dict.
    """
    doc = json.loads(Path(path).expanduser().read_text())
    if isinstance(doc, dict) and isinstance(doc.get("stats"), dict):
        return doc["stats"]
    return doc if isinstance(doc, dict) else {}


# ---------------------------------------------------------------------------------------------
# Scoring core (pure, deterministic).
# ---------------------------------------------------------------------------------------------


def robust_z(value, stat: dict) -> Optional[float]:
    """Domain-aware robust-z of ``value`` against one key's stats, or None if not scorable.

    ``z = (t(value) − median) / MAD`` where ``t`` is ``log10`` for a ``log10``-domain key and
    identity for a ``linear`` key. Returns None for a non-positive value in a log key, or when
    the stat lacks a median/MAD. |z| is capped at :data:`Z_CAP`.
    """
    if value is None or not isinstance(stat, dict):
        return None
    median = stat.get("median")
    mad = stat.get("mad")
    if median is None or mad is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    domain = stat.get("domain", "linear")
    if domain == "log10":
        if v <= 0.0:
            return None
        v = math.log10(v)
    mad_eff = mad if (mad and mad > MAD_FLOOR) else MAD_FLOOR
    z = (v - float(median)) / mad_eff
    return max(-Z_CAP, min(Z_CAP, z))


def axis_band(z: float) -> str:
    """Per-axis verdict label from a signed z: in_band / below / above / far_below / far_above."""
    az = abs(z)
    if az < Z_IN_BAND:
        return "in_band"
    direction = "above" if z > 0 else "below"
    return direction if az < Z_FAR else f"far_{direction}"


def verdict_word(max_abs_z: float) -> str:
    """Batch verdict from the worst axis: PASS (<2) / ALTER (2–3.5) / REJECT (≥3.5)."""
    if max_abs_z < Z_IN_BAND:
        return "PASS"
    return "ALTER" if max_abs_z < Z_FAR else "REJECT"


def verdict_color(max_abs_z: float) -> str:
    return _COLOR[verdict_word(max_abs_z)]


def short_key(key: str) -> str:
    """A compact axis name for burned-in text (``spectrum.band.sub`` → ``sub``)."""
    parts = key.split(".")
    if len(parts) >= 3 and parts[0] in ("spectrum", "width"):
        # e.g. spectrum.band.sub → "sub"; width.sub.correlation → "sub.corr"
        if parts[1] in ("band", "oct6"):
            return parts[2]
        tail = parts[2].replace("correlation", "corr").replace("side_mid_ratio", "smr")
        return f"{parts[1]}.{tail}"
    return parts[-1]


def score_candidate(feat: dict, profile: dict) -> dict:
    """Score one candidate's flat feature dict against a per-key profile.

    Returns ``{per_key, agg, max_abs, worst, n_keys}`` where ``agg`` is the mean |z| over the
    profiled keys the candidate actually carries (the ranking key; ``inf`` when none overlap),
    ``max_abs`` is the worst axis (the verdict driver), and ``worst`` is the axes ordered by
    descending |z| as ``[(key, z, axis_band)]``.
    """
    per: dict[str, float] = {}
    for key, stat in profile.items():
        if key not in feat:
            continue
        z = robust_z(feat[key], stat)
        if z is not None:
            per[key] = z
    if not per:
        return {"per_key": {}, "agg": float("inf"), "max_abs": float("inf"),
                "worst": [], "n_keys": 0}
    abs_z = {k: abs(v) for k, v in per.items()}
    agg = sum(abs_z.values()) / len(abs_z)
    max_abs = max(abs_z.values())
    ordered = sorted(per.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    worst = [(k, v, axis_band(v)) for k, v in ordered]
    return {"per_key": per, "agg": agg, "max_abs": max_abs, "worst": worst, "n_keys": len(per)}


def candidate_label(audio_frame: dict, index: int) -> str:
    """A human/LLM label for a candidate tile (meta.name → source basename → role → cand-N)."""
    meta = audio_frame.get("meta") or {}
    if meta.get("name"):
        return str(meta["name"])
    params = audio_frame.get("params") or {}
    src = params.get("source") or params.get("name")
    if src:
        return Path(str(src)).name
    role = audio_frame.get("role")
    if role and role not in ("source", None):
        return str(role)
    h = audio_frame.get("hash") or ""
    return h.split(":")[-1][:8] if h else f"cand-{index}"


def candidate_features(frames: list[dict], audio_id: str) -> dict:
    """Flatten every `feature` frame tagged to ``audio_id`` into one ``{key: scalar}`` dict."""
    flat: dict[str, float] = {}
    for f in frames:
        if f.get("kind") != "feature" or f.get("of") != audio_id:
            continue
        data = f.get("data")
        if not isinstance(data, dict):
            continue
        for key, val in data.items():
            scalar = _scalar_of(val)
            if scalar is not None:
                flat[key] = scalar
    return flat


def rank_candidates(frames: list[dict], profile: dict, *, role: Optional[str] = None) -> list[dict]:
    """Rank the batch's candidate audio frames best→worst by aggregate robust-z.

    Deterministic: sorted by (agg |z| ascending, label). Returns a list of entries
    ``{frame, label, score, index}``.
    """
    audios = [f for f in frames
              if f.get("kind") == "audio" and f.get("hash")
              and (role is None or f.get("role") == role)]
    scored = []
    for i, af in enumerate(audios):
        feat = candidate_features(frames, af["id"])
        scored.append({
            "frame": af, "label": candidate_label(af, i),
            "score": score_candidate(feat, profile), "index": i,
        })
    scored.sort(key=lambda e: (e["score"]["agg"], e["label"]))
    return scored


# ---------------------------------------------------------------------------------------------
# Rendering helpers.
# ---------------------------------------------------------------------------------------------


def _use_agg():
    import matplotlib

    matplotlib.use("Agg")


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _tile_text(entry: dict) -> str:
    """Burned-in tile caption: ``label`` / ``VERDICT · worst-axis band[ · 2nd-axis band]``."""
    sc = entry["score"]
    label = entry["label"]
    if sc["n_keys"] == 0:
        return f"{label}\n(no profiled features)"
    verdict = verdict_word(sc["max_abs"])
    worst = sc["worst"][:2]
    axes = " · ".join(f"{short_key(k)} {band}" for k, _z, band in worst)
    return f"{label}\n{verdict} · {axes}"


def render_contact_sheet(
    frames: list[dict],
    profile: dict,
    *,
    role: Optional[str] = None,
    cols: Optional[int] = None,
    title: Optional[str] = None,
) -> list[dict]:
    """Render the batch contact sheet; return ``[image_frame, text_index_frame]``.

    Emits nothing (``[]``) when the batch has no candidate audio frames. The image frame's
    ``params.order`` carries the deterministic best→worst ranking so a consumer never re-sorts.
    The text index is the "scalar strip" — the same ranking as markdown (0 image tokens).
    """
    import numpy as np

    from smplstream import cas, frames as F

    ranked = rank_candidates(frames, profile, role=role)
    if not ranked:
        return []

    _use_agg()
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from .spectrogram import pad_short_signal

    # --- load every candidate's mel power once; calibrate one dB scale across the batch. ---
    mels: list[Optional[tuple]] = []  # (S_power, sr) or None on failure
    global_max = 1e-12
    for entry in ranked:
        try:
            src = cas.get_path(entry["frame"]["hash"])
            y, sr = librosa.load(str(src), sr=None, mono=True)
            y = pad_short_signal(y, _N_FFT)  # short tiles → full window (real mel, not "no audio")
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=_N_FFT, hop_length=_HOP, n_mels=_N_MELS
            )
            mels.append((S, int(sr)))
            if S.size:
                global_max = max(global_max, float(np.max(S)))
        except Exception:
            mels.append(None)

    norm = Normalize(vmin=-_TOP_DB, vmax=0.0)  # shared dB scale for every tile + colorbar
    n = len(ranked)
    ncols = cols or min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.3, nrows * 3.0), squeeze=False)

    mappable = None
    for idx, entry in enumerate(ranked):
        ax = axes[idx // ncols][idx % ncols]
        cell = mels[idx]
        color = verdict_color(entry["score"]["max_abs"]) if entry["score"]["n_keys"] else "#607d8b"
        if cell is not None:
            S, sr = cell
            S_db = librosa.power_to_db(S, ref=global_max)
            img = librosa.display.specshow(
                S_db, sr=sr, hop_length=_HOP, x_axis="time", y_axis="mel",
                ax=ax, cmap="magma", norm=norm,
            )
            mappable = mappable or img
            for hz in _GRIDLINES:
                if hz < sr / 2.0:
                    ax.axhline(hz, color="white", alpha=0.20, linewidth=0.6)
        else:
            ax.text(0.5, 0.5, "no audio", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=5)
        ax.text(
            0.02, 0.97, _tile_text(entry), transform=ax.transAxes, va="top", ha="left",
            fontsize=6.5, color="white", family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3.0)
            spine.set_visible(True)

    # Blank any unused grid cells.
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), format="%+2.0f dB",
                            fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
    head = title or f"Contact sheet — {n} candidates, sorted best→worst (robust-z vs profile)"
    fig.suptitle(head + "  ·  green PASS / amber ALTER / red REJECT", fontsize=9)
    png = _fig_to_png(fig)
    plt.close(fig)

    h = cas.put_blob(png, "image/png")
    order = [e["label"] for e in ranked]
    scores = [
        {
            "label": e["label"],
            "agg": None if math.isinf(e["score"]["agg"]) else round(e["score"]["agg"], 4),
            "max_abs": None if math.isinf(e["score"]["max_abs"]) else round(e["score"]["max_abs"], 4),
            "verdict": verdict_word(e["score"]["max_abs"]) if e["score"]["n_keys"] else "UNSCORED",
            "worst": [[short_key(k), round(z, 3), band] for k, z, band in e["score"]["worst"][:2]],
            "n_keys": e["score"]["n_keys"],
        }
        for e in ranked
    ]
    image = F.image_frame(
        h, media="image/png", role="contact-sheet", op=OP_SHEET, op_version=OP_SHEET_VERSION,
        params={"order": order, "scores": scores, "n": n,
                "calibration": {"vmin": -_TOP_DB, "vmax": 0.0, "ref": "batch_max"}},
        meta={"cols": ncols, "rows": nrows},
    )

    # The "scalar strip" — the deterministic ranking as markdown (the A/B alternative to tiles).
    lines = ["| # | candidate | verdict | agg \\|z\\| | worst axes |",
             "|---|---|---|---|---|"]
    for i, s in enumerate(scores, 1):
        worst = ", ".join(f"{k} {band} (z={z})" for k, z, band in s["worst"]) or "—"
        agg = "—" if s["agg"] is None else f"{s['agg']:.2f}"
        lines.append(f"| {i} | {s['label']} | {s['verdict']} | {agg} | {worst} |")
    index = F.text_frame(
        "\n".join(lines), role="contact-sheet-index", op=OP_SHEET, op_version=OP_SHEET_VERSION,
        params={"order": order},
    )
    return [image, index]


def render_profile_overlay(audio_frame: dict, profile: dict) -> list[dict]:
    """Render the candidate spectrum drawn over the role profile; return ``[image_frame]``.

    The candidate's 1/6-octave spectrum (dB, total-power reference) is drawn over the profile
    median with a shaded p10–p90 band (when the stats file carries ``spectrum.oct6.*`` keys),
    and the six band-deltas (candidate − profile median, from ``spectrum.band.*``) are printed
    as text. Renders a valid PNG even when the stats file carries no spectrum keys.
    """
    import numpy as np
    import soundfile as sf

    from smplstream import cas, frames as F

    from . import octave as OCT

    src = cas.get_path(audio_frame["hash"])
    samples, sr = sf.read(str(src), dtype="float32", always_2d=True)
    y = samples.T

    centers, db = OCT.octave_spectrum(y, sr)
    cand_bands = OCT.band_levels(y, sr)

    # --- pull the profile spectrum curve (median + p10/p90) from the stats keys. ---
    prof_pts = []
    for key, stat in profile.items():
        if not key.startswith("spectrum.oct6.") or not isinstance(stat, dict):
            continue
        try:
            hz = float(key.rsplit(".", 1)[1])
        except (ValueError, IndexError):
            continue
        if stat.get("median") is None:
            continue
        prof_pts.append((hz, stat.get("median"), stat.get("p10"), stat.get("p90")))
    prof_pts.sort(key=lambda t: t[0])
    has_profile = len(prof_pts) >= 2

    _use_agg()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.semilogx(centers, db, color="#1565c0", linewidth=2.0, label="candidate", zorder=3)
    if has_profile:
        pc = np.asarray([p[0] for p in prof_pts])
        pmed = np.asarray([float(p[1]) for p in prof_pts])
        ax.semilogx(pc, pmed, color="#37474f", linewidth=1.6, linestyle="--",
                    label="profile median", zorder=2)
        p10 = np.asarray([p[2] if p[2] is not None else p[1] for p in prof_pts], dtype="float64")
        p90 = np.asarray([p[3] if p[3] is not None else p[1] for p in prof_pts], dtype="float64")
        ax.fill_between(pc, p10, p90, color="#90a4ae", alpha=0.30, label="profile p10–p90", zorder=1)

    for hz in _GRIDLINES:
        if hz < sr / 2.0:
            ax.axvline(hz, color="grey", alpha=0.30, linewidth=0.7)

    ax.set_xlim(OCT.FMIN, min(OCT.FMAX, sr / 2.0))
    ax.set_xlabel("Frequency (Hz, log)")
    ax.set_ylabel("Level (dB rel. total)")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title("Spectrum vs profile" + ("" if has_profile else "  (no spectrum profile in stats)"))

    # --- six band-deltas as burned-in text. ---
    band_deltas: dict[str, Optional[float]] = {}
    lines = ["band Δ (cand − profile med), dB:"]
    for name, _lo, _hi in BANDS:
        if name == "full":
            continue
        cand = cand_bands.get(name)
        pstat = profile.get(f"spectrum.band.{name}")
        pmed = pstat.get("median") if isinstance(pstat, dict) else None
        if cand is not None and pmed is not None:
            delta = round(float(cand) - float(pmed), 2)
            band_deltas[name] = delta
            lines.append(f"  {name:<8} {delta:+.2f}")
        elif cand is not None:
            band_deltas[name] = None
            lines.append(f"  {name:<8} {cand:+.1f} (no ref)")
    ax.text(
        0.99, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="right",
        fontsize=7, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.75, edgecolor="#bbb"),
    )

    png = _fig_to_png(fig)
    plt.close(fig)

    h = cas.put_blob(png, "image/png")
    return [
        F.image_frame(
            h, media="image/png", role="profile-overlay", of=audio_frame["id"],
            op=OP_OVERLAY, op_version=OP_OVERLAY_VERSION,
            params={"has_profile": has_profile, "n_bins": int(len(centers)),
                    "band_deltas": band_deltas},
            meta={"sr": int(sr)},
        )
    ]
