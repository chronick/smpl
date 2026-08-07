"""Per-band stereo width / correlation (ticket vault-3tuy).

Broadband width hides the failure that actually matters on a club stack: a *wide sub* that
mono-collapses (cancels) on a big mono system. A single full-band correlation number can look
healthy while the sub is anti-phased. So width is measured **per frequency band** as well as
full-band.

For each band the signal is band-limited (zero-phase Butterworth) and two numbers come out:

  ``correlation``    — Pearson L/R correlation in the band, −1..1 (mono-compatibility;
                       +1 = mono, 0 = decorrelated/wide, −1 = anti-phase → cancels in mono)
  ``side_mid_ratio`` — RMS(side) / RMS(mid) with mid = (L+R)/2, side = (L−R)/2
                       (0 = mono, grows with stereo width)

The six standardized bands (edges registered in feature-keys.md):

  Sub 20–60 · Bass 60–200 · LoMid 200–500 · Mid 500–2k · UpperMid 2–6k · Air 6–20k

plus a ``full`` band (20 Hz–Nyquist) for the broadband reference. Emitted keys are
``width.<band>.correlation`` / ``width.<band>.side_mid_ratio``.

Mono input (1 channel, or identical L/R): correlation = 1.0, side_mid_ratio = 0.0 for every
band — a mono file is trivially mono-compatible and zero-width by construction.

Pure functions returning frame dicts. Heavy imports stay inside the functions.
"""

from __future__ import annotations

OP = "width"
OP_VERSION = "width@1"

# The six standardized band edges (Hz), plus the broadband reference. Order is stable so the
# emitted key set is deterministic. ``None`` upper edge = clamp to just under Nyquist.
BANDS: list[tuple[str, float, float]] = [
    ("sub", 20.0, 60.0),
    ("bass", 60.0, 200.0),
    ("lomid", 200.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("uppermid", 2000.0, 6000.0),
    ("air", 6000.0, 20000.0),
    ("full", 20.0, 20000.0),
]

_EPS = 1e-12


def _bandpass(x, sr: int, lo: float, hi: float, order: int = 4):
    """Zero-phase Butterworth band-pass restricted to the valid (0, Nyquist) range."""
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, dtype="float64")
    nyq = sr / 2.0
    lo_w = max(lo / nyq, 1e-4)
    hi_w = min(hi / nyq, 0.999)
    if hi_w <= lo_w or x.shape[0] <= 3 * order * 2:
        return x
    sos = butter(order, [lo_w, hi_w], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def _corr_and_width(left, right) -> tuple:
    """(Pearson L/R correlation, side/mid RMS ratio) for a stereo band."""
    import numpy as np

    left = np.asarray(left, dtype="float64")
    right = np.asarray(right, dtype="float64")
    ls = left - left.mean()
    rs = right - right.mean()
    denom = float(np.sqrt(np.sum(ls * ls) * np.sum(rs * rs)))
    corr = float(np.clip(np.sum(ls * rs) / denom, -1.0, 1.0)) if denom > 0.0 else 1.0

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_rms = float(np.sqrt(np.mean(mid * mid)))
    side_rms = float(np.sqrt(np.mean(side * side)))
    ratio = side_rms / mid_rms if mid_rms > _EPS else 0.0
    return corr, ratio


def per_band_width(samples, sr: int) -> dict:
    """Per-band stereo width/correlation for a ``(n, ch)`` float array.

    Returns a flat dict of ``width.<band>.correlation`` / ``width.<band>.side_mid_ratio``
    keys for every band in :data:`BANDS`. Mono (or effectively mono) input yields
    correlation 1.0 / ratio 0.0 for all bands.
    """
    import numpy as np

    s = np.asarray(samples, dtype="float64")
    if s.ndim == 1:
        s = s[:, None]

    out: dict = {}
    if s.shape[1] < 2:  # mono → trivially mono-compatible, zero width
        for name, _lo, _hi in BANDS:
            out[f"width.{name}.correlation"] = 1.0
            out[f"width.{name}.side_mid_ratio"] = 0.0
        return out

    left = s[:, 0]
    right = s[:, 1]
    for name, lo, hi in BANDS:
        lb = _bandpass(left, sr, lo, hi)
        rb = _bandpass(right, sr, lo, hi)
        corr, ratio = _corr_and_width(lb, rb)
        out[f"width.{name}.correlation"] = round(corr, 4)
        out[f"width.{name}.side_mid_ratio"] = round(ratio, 4)
    return out


def width_audio_frame(audio_frame: dict) -> list[dict]:
    """Resolve an `audio` frame's PCM and emit its per-band width feature frame.

    Returns a one-element list with the derived `feature` frame (role ``width``), carrying
    ``of``/``op``/``op_version``/``params`` lineage. The caller passes the input through.
    """
    import soundfile as sf

    from smplstream import cas, frames as F

    src = cas.get_path(audio_frame["hash"])
    samples, sr = sf.read(str(src), dtype="float32", always_2d=True)  # (n, ch)

    data = per_band_width(samples, sr)
    params = {"bands": {name: [lo, hi] for name, lo, hi in BANDS}, "sr": int(sr),
              "ch": int(samples.shape[1])}
    return [
        F.feature_frame(
            data, role="width", of=audio_frame["id"], op=OP, op_version=OP_VERSION,
            params=params,
        )
    ]
