"""`describe` — the light-tier aggregator (the op behind `smpl cat` / `smpl describe`).

`smpl cat` delegates to :func:`describe_audio_frame` when this module is importable
(see `smpl_cli/subcommands/cat.py`). One audio frame in → a *list* of derived frames
out, aggregating the whole light analysis tier in one pass:

  - **loudness** (`smpl_analysis.loudness`) — BS.1770 integrated/true-peak/short-term
  - **spectral shape** (`smpl_analysis.spectral`) — flatness/crest/spread/rolloff/…
  - **technical QC** (`smpl_analysis.qc`) — clipping/phase/DC/SNR/lossy-origin (+ defect markers)
  - **one mel spectrogram** (`smpl_analysis.spectrogram`) — when `want_image`
  - **a caption** (`text` frame, role `caption`) — a concise human/LLM summary of the
    headline numbers (duration, sr, ch, integrated LUFS, true-peak, brightness, flatness,
    and any QC flags), synthesized from the feature frames above.

The through-line (research §0): *if it doesn't need reasoning, it shouldn't call a model* —
this tier is pure deterministic DSP; the LLM's job is to interpret these frames, not
produce them. The caption is a deterministic template, not a model call.

**Resilience contract.** Each sub-analysis runs in isolation: if one raises, we append an
`error` frame (`code: op_failed`) tagged to the audio id and keep going — describe never
aborts the whole aggregation because one tier failed. Every derived frame carries
`of`/`op`/`op_version` lineage per the tool contract; the caller (`smpl cat`) is
responsible for passthrough of the input audio frame.

This module emits **no new feature keys** of its own — it reuses the keys the sub-tiers
already own (feature-keys.md). The caption is a `text` frame (prose), not a `feature`.
"""

from __future__ import annotations

import math
from typing import Any, Optional

OP = "describe"
OP_VERSION = "describe@1"


def _fmt_num(x: Any, unit: str = "", *, ndigits: int = 1) -> str:
    """Render a possibly-None/NaN number for the caption, with an optional unit suffix."""
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        return "n/a"
    s = f"{xf:.{ndigits}f}"
    return f"{s} {unit}".rstrip() if unit else s


def _feature_data(frames: list[dict], role: str) -> dict:
    """Pull the `data` dict of the (last-wins) feature frame with the given role."""
    hit: dict = {}
    for f in frames:
        if f.get("kind") == "feature" and f.get("role") == role:
            data = f.get("data")
            if isinstance(data, dict):
                hit = data
    return hit


def _stat_mean(value: Any) -> Optional[float]:
    """Unwrap a `{mean, stdev}` stat object (or a bare scalar) to its mean."""
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def synthesize_caption(audio_frame: dict, derived: list[dict]) -> str:
    """Build the human/LLM caption string from the audio frame + the derived feature frames.

    Reads the loudness/spectral/qc feature frames already produced this pass (so the caption
    never recomputes anything) and renders a one-line-ish summary of the headline numbers.
    Resilient to any sub-tier having been skipped — missing values render as `n/a` and QC
    flags only appear when present.
    """
    meta = audio_frame.get("meta") or {}
    sr = meta.get("sr")
    ch = meta.get("ch")
    dur = meta.get("dur")

    loud = _feature_data(derived, "loudness")
    spec = _feature_data(derived, "spectral")
    qc = _feature_data(derived, "qc")

    integrated = loud.get("loudness.integrated_lufs")
    true_peak = loud.get("loudness.true_peak_dbtp")

    # Brightness proxy: spectral rolloff (Hz) mean — the high-frequency energy edge.
    brightness_hz = _stat_mean(spec.get("lowlevel.spectral_rolloff"))
    flatness_db = _stat_mean(spec.get("lowlevel.spectral_flatness_db"))

    head = (
        f"{_fmt_num(dur, 's', ndigits=2)} · "
        f"{sr if sr is not None else 'n/a'} Hz · "
        f"{ch if ch is not None else 'n/a'}ch"
    )
    loud_part = (
        f"integrated {_fmt_num(integrated, 'LUFS')} · "
        f"true-peak {_fmt_num(true_peak, 'dBTP')}"
    )
    timbre_part = (
        f"brightness ~{_fmt_num(brightness_hz, 'Hz', ndigits=0)} · "
        f"flatness {_fmt_num(flatness_db, 'dB')}"
    )

    parts = [head, loud_part, timbre_part]

    # QC flags — only surface the ones that actually fired (clean by omission).
    flags: list[str] = []
    if qc:
        if qc.get("qc.clipping.detected"):
            flags.append("clipping")
        corr = qc.get("qc.phase.correlation")
        if isinstance(corr, (int, float)) and corr < 0.0:
            flags.append(f"phase {corr:+.2f}")
        if qc.get("qc.lossy.confidence", 0) and qc["qc.lossy.confidence"] >= 0.5:
            cutoff = qc.get("qc.lossy.spectral_cutoff_hz")
            flags.append(f"lossy? cutoff ~{_fmt_num(cutoff, 'Hz', ndigits=0)}")
        snr = qc.get("qc.snr_db")
        if isinstance(snr, (int, float)) and snr < 20.0:
            flags.append(f"low SNR {snr:.0f} dB")
    if flags:
        parts.append("QC: " + ", ".join(flags))

    return " — ".join(parts)


def describe_audio_frame(audio_frame: dict, *, want_image: bool = True) -> list[dict]:
    """Aggregate the light analysis tier for one `audio` frame; return derived frames.

    Returns a list of derived frames (no passthrough — the caller passes the input
    through first). Order: loudness → spectral → qc feature/marker frames, then one mel
    `image` frame (when `want_image`), then a `text` caption frame summarizing them.

    Each sub-analysis is isolated: a failure in one tier yields an `error` frame
    (`op_failed`, tagged to the audio id and the failing sub-op) and the rest continue.
    The caption is always emitted, synthesized from whatever feature frames succeeded.
    """
    from smplstream import error_frame

    from . import loudness as L, qc as QC, spectral as SP

    of = audio_frame.get("id")
    out: list[dict] = []

    # --- loudness tier (vault-3vau) ---
    try:
        out.extend(L.loudness_frames(audio_frame))
    except Exception as exc:  # one tier failing must not abort the aggregation
        out.append(error_frame("op_failed", f"loudness: {exc}", of=of, op=OP))

    # --- spectral-shape tier (vault-3uap) ---
    try:
        out.extend(SP.spectral_audio_frame(audio_frame))
    except Exception as exc:
        out.append(error_frame("op_failed", f"spectral: {exc}", of=of, op=OP))

    # --- technical QC tier (vault-1e9a) ---
    try:
        out.extend(QC.qc_audio_frame(audio_frame))
    except Exception as exc:
        out.append(error_frame("op_failed", f"qc: {exc}", of=of, op=OP))

    # --- one mel spectrogram image for the vision LLM (research §2) ---
    if want_image:
        try:
            from . import spectrogram as IMG

            out.extend(IMG.render_audio_frame(audio_frame, kinds=["mel"]))
        except Exception as exc:
            out.append(error_frame("op_failed", f"spectrogram: {exc}", of=of, op=OP))

    # --- caption (deterministic prose over the numbers above; never a model call) ---
    try:
        from smplstream import frames as F

        caption = synthesize_caption(audio_frame, out)
        out.append(
            F.text_frame(
                caption,
                role="caption",
                of=of,
                op=OP,
                op_version=OP_VERSION,
                params={"want_image": bool(want_image)},
            )
        )
    except Exception as exc:
        out.append(error_frame("op_failed", f"caption: {exc}", of=of, op=OP))

    return out
