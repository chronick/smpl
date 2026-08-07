"""Fractional-octave spectrum feature family (ticket vault-22oy).

A level-invariant description of a sample's tonal SHAPE: a Welch PSD binned into
**1/6-octave bins** (dB, normalized to total in-band power so two takes at different
loudness compare on shape alone), plus the six standardized band levels.

Two purposes, one DSP:

  1. It is the candidate curve the ``profile-overlay`` viz draws.
  2. Emitted as a `feature` frame (``smpl octave-spectrum``) it flows into
     ``smpl stats build`` exactly like any other feature — reducing a role corpus to a
     median spectrum + p10/p90 band the overlay then draws behind the candidate. No roles
     or thresholds are baked in here (the layering rule): this op emits measurements only.

Emitted keys:

  ``spectrum.oct6.<center_hz>``   dB of the 1/6-octave bin centered near <center_hz>,
                                  relative to total in-band power (≤ 0 dB).
  ``spectrum.band.<name>``        dB of each standardized band (sub/bass/lomid/mid/
                                  uppermid/air), same total-power reference — the six
                                  band-deltas the overlay prints.

Pure functions returning frame dicts. Heavy imports (numpy/scipy) live inside functions so
cold pipe stages start fast.
"""

from __future__ import annotations

from typing import Optional

from .width import BANDS  # single source of truth for the six standardized band edges

OP = "octave-spectrum"
OP_VERSION = "octave-spectrum@1"

# --- pinned recipe constants (declared so memo params are complete & stable per op_version) --
BINS_PER_OCTAVE = 6         # 1/6-octave resolution
FMIN = 20.0                 # low edge of the analyzed range (Hz)
FMAX = 20000.0              # high edge (clamped to just under Nyquist)
NPERSEG = 4096              # Welch segment length (clamped to signal length)
REF_HZ = 1000.0            # octave-grid reference (nominal 1/6-octave centers)

_FLOOR_DB = -120.0
_EPS = 1e-20


def _to_mono(y):
    import numpy as np

    y = np.asarray(y, dtype="float64")
    if y.ndim > 1:  # accept (ch, n) or (n, ch); collapse the channel axis
        y = y.mean(axis=0) if y.shape[0] < y.shape[1] else y.mean(axis=1)
    return np.ascontiguousarray(y)


def _welch(y, sr: int):
    """Welch one-sided PSD ``(freqs, power)`` for a mono signal (robust to short input)."""
    import numpy as np
    from scipy.signal import welch

    mono = _to_mono(y)
    if mono.size < 8:
        # Too short for a meaningful PSD — a single flat bin keeps the pipeline alive.
        return np.asarray([0.0, sr / 2.0]), np.asarray([_EPS, _EPS])
    nperseg = int(min(NPERSEG, mono.size))
    f, pxx = welch(mono, fs=sr, nperseg=nperseg)
    return np.asarray(f, dtype="float64"), np.asarray(pxx, dtype="float64")


def _octave_centers(fmax_eff: float) -> list[float]:
    """Nominal 1/6-octave center frequencies within ``[FMIN, fmax_eff]`` (ref 1 kHz)."""
    import numpy as np

    n_lo = int(np.ceil(BINS_PER_OCTAVE * np.log2(FMIN / REF_HZ)))
    n_hi = int(np.floor(BINS_PER_OCTAVE * np.log2(fmax_eff / REF_HZ)))
    return [REF_HZ * (2.0 ** (n / BINS_PER_OCTAVE)) for n in range(n_lo, n_hi + 1)]


def _band_power(f, pxx, lo: float, hi: float) -> float:
    """Summed PSD power in ``[lo, hi)`` (linear)."""
    import numpy as np

    mask = (f >= lo) & (f < hi)
    return float(np.sum(pxx[mask]))


def _to_db(power: float, total: float) -> float:
    import numpy as np

    ratio = power / total if total > _EPS else 0.0
    return float(max(10.0 * np.log10(ratio + _EPS), _FLOOR_DB))


def octave_spectrum(y, sr: int, *, bpo: int = BINS_PER_OCTAVE):
    """Return ``(centers_hz, db)`` — the level-normalized 1/6-octave spectrum of a signal.

    dB is relative to the total in-band ([FMIN, Nyquist]) power, so the curve describes
    spectral SHAPE independent of absolute loudness. ``centers_hz`` is strictly ascending.
    """
    import numpy as np

    f, pxx = _welch(y, sr)
    nyq = sr / 2.0
    fmax_eff = min(FMAX, nyq * 0.99)
    total = float(np.sum(pxx[(f >= FMIN) & (f <= fmax_eff)]))

    centers = _octave_centers(fmax_eff)
    half = 2.0 ** (1.0 / (2.0 * bpo))
    db = [_to_db(_band_power(f, pxx, c / half, c * half), total) for c in centers]
    return np.asarray(centers, dtype="float64"), np.asarray(db, dtype="float64")


def band_levels(y, sr: int) -> dict:
    """The six standardized band levels (dB, total-power reference): ``{name: db}``."""
    import numpy as np

    f, pxx = _welch(y, sr)
    nyq = sr / 2.0
    fmax_eff = min(FMAX, nyq * 0.99)
    total = float(np.sum(pxx[(f >= FMIN) & (f <= fmax_eff)]))
    out: dict[str, float] = {}
    for name, lo, hi in BANDS:
        if name == "full":
            continue
        out[name] = _to_db(_band_power(f, pxx, lo, min(hi, fmax_eff)), total)
    return out


def octave_spectrum_scalars(y, sr: int) -> dict:
    """Flat feature dict: 1/6-octave bins + six band levels (dB, total-power reference)."""
    centers, db = octave_spectrum(y, sr)
    out = {f"spectrum.oct6.{int(round(c))}": round(float(d), 3) for c, d in zip(centers, db)}
    for name, d in band_levels(y, sr).items():
        out[f"spectrum.band.{name}"] = round(float(d), 3)
    return out


def octave_audio_frame(audio_frame: dict) -> list[dict]:
    """Resolve an `audio` frame's PCM and emit its 1/6-octave spectrum feature frame.

    Returns a one-element list with the derived `feature` frame (role ``octave-spectrum``),
    carrying ``of``/``op``/``op_version``/``params`` lineage. The caller passes the input
    frame through.
    """
    import soundfile as sf

    from smplstream import cas, frames as F

    src = cas.get_path(audio_frame["hash"])
    samples, sr = sf.read(str(src), dtype="float32", always_2d=True)  # (n, ch)

    data = octave_spectrum_scalars(samples.T, sr)
    params = {
        "bins_per_octave": BINS_PER_OCTAVE, "fmin": FMIN, "fmax": FMAX,
        "nperseg": NPERSEG, "ref_hz": REF_HZ, "sr": int(sr),
        "normalization": "total_inband_power_db",
    }
    return [
        F.feature_frame(
            data, role="octave-spectrum", of=audio_frame["id"], op=OP, op_version=OP_VERSION,
            params=params,
        )
    ]
