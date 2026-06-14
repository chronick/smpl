"""Technical QC / defect detection (research §4; ticket vault-1e9a).

The "is this sample technically clean and usable" axis — the deterministic DSP top-6 that
curators and engineers reject for. Pure functions over a resolved audio path / np array
returning smplstream frame dicts:

  1. clipping / true-peak     → `qc.clipping.detected` pass/fail, keyed off an ITU-R BS.1770
                                 true-peak we compute internally; the `loudness.true_peak_dbtp`
                                 dBTP *measurement* is owned by the loudness tier, so we do NOT
                                 re-emit it under a qc.* key (one measurement, one owner)
  2. phase correlation        → `qc.phase.correlation` (−1..1, stereo mono-compatibility)
  3. DC offset                → `qc.dc_offset_dbfs`
  4. noise floor / SNR        → `qc.snr_db`
  5. clicks / gaps            → `marker` frames at defect locations
  6. lossy-origin via cutoff  → `qc.lossy.spectral_cutoff_hz`, `qc.lossy.expected_nyquist_hz`,
                                 `qc.lossy.confidence` (average-FFT brickwall detector)

All measurement keys come from the registry (feature-keys.md, owner vault-1e9a). Heavy
imports (numpy/scipy/soundfile) stay inside functions so cold pipe stages start fast.
"""

from __future__ import annotations

from typing import Any, Optional

OP = "qc"
OP_VERSION = "qc@1"

# A sample at or above this absolute true-peak (in dBTP) is flagged as clipping.
_CLIP_DBTP_THRESHOLD = -0.1
# Fraction of the spectral energy that must fall below the cutoff for it to count as a
# brickwall (LAME/MP3 low-passes hard, so almost all energy sits below the knee).
_LOSSY_ENERGY_FRAC = 0.999
# How far below Nyquist the knee must sit (fraction) before we even consider a brickwall —
# full-band content has its cutoff right at Nyquist and must score ~0.
_LOSSY_MIN_KNEE_FRAC = 0.05
# Confidence saturates once the band well above the knee is this many dB below the in-band
# level (a real LAME/AAC brickwall buries the upper band at the noise floor: −60 dB+).
_LOSSY_FLOOR_DB = 60.0


def _dbfs(x: float) -> float:
    import math

    return 20.0 * math.log10(x) if x > 0 else float("-inf")


def _load(path: str):
    """Decode to (samples [n, ch] float32, sr). No resampling / normalization."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, int(sr)


def true_peak_dbtp(samples, sr: int) -> float:
    """ITU-R BS.1770 true-peak estimate via 4x polyphase oversampling, in dBTP.

    Per-channel max of the oversampled signal; the overall true-peak is the channel max.
    We compute it here only to key the clipping flag — the canonical `loudness.true_peak_dbtp`
    feature key stays owned by the loudness tier (vault-3vau).
    """
    import numpy as np
    from scipy.signal import resample_poly

    if samples.size == 0:
        return float("-inf")
    peak = 0.0
    for ch in range(samples.shape[1]):
        up = resample_poly(samples[:, ch].astype(np.float64), 4, 1)
        peak = max(peak, float(np.max(np.abs(up))) if up.size else 0.0)
    # also consider the raw sample peak (oversampling only ever raises it, but guard size==1)
    peak = max(peak, float(np.max(np.abs(samples))))
    return _dbfs(peak)


def phase_correlation(samples) -> Optional[float]:
    """Inter-channel (L/R) Pearson correlation in [−1, 1]. None for mono / degenerate.

    +1 = perfectly mono-compatible (in phase); 0 = decorrelated/wide; −1 = anti-phase
    (cancels to silence in mono). Computed on the first two channels.
    """
    import numpy as np

    if samples.shape[1] < 2:
        return None
    left = samples[:, 0].astype(np.float64)
    right = samples[:, 1].astype(np.float64)
    ls = left - left.mean()
    rs = right - right.mean()
    denom = float(np.sqrt(np.sum(ls * ls) * np.sum(rs * rs)))
    if denom <= 0.0:
        return None
    return float(np.clip(np.sum(ls * rs) / denom, -1.0, 1.0))


def dc_offset_dbfs(samples) -> float:
    """Worst-channel DC offset (mean sample value) expressed in dBFS."""
    import numpy as np

    if samples.size == 0:
        return float("-inf")
    means = np.abs(samples.mean(axis=0))
    return _dbfs(float(np.max(means)))


def snr_db(samples, sr: int) -> Optional[float]:
    """Crude dynamic-range SNR: loud-passage RMS vs the quietest-passage RMS (the noise floor).

    Frame the mono mix and take per-frame RMS. The signal level is the 90th-percentile frame
    RMS; the noise floor is the median of the quietest 5% of frames. SNR = signal − noise in
    dB, clamped to a 120 dB ceiling.

    This is the cheap "how far does the body sit above the quiet/silent regions" measure —
    it is most meaningful for material that HAS quiet passages (decay, gaps, head/tail
    silence), where the floor is the actual noise. A perfectly steady continuous signal with
    no quiet region reports a small value by construction (every frame is equally loud — there
    is no quiet passage to contrast against); that is a correct statement about the signal, not
    a hidden failure. Returns None only when the clip is too short to frame or is pure digital
    silence.
    """
    import numpy as np

    if samples.size == 0:
        return None
    mono = samples.mean(axis=1).astype(np.float64)
    win = max(256, int(sr * 0.02))  # ~20 ms frames
    hop = win // 2
    if mono.shape[0] < win:
        return None
    n_frames = 1 + (mono.shape[0] - win) // hop
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = mono[i * hop : i * hop + win]
        rms[i] = float(np.sqrt(np.mean(seg * seg)))
    # Keep silent (zero-RMS) frames: they ARE the noise floor we want to measure against.
    # Dropping them would measure the floor over the loud body and collapse SNR to ~0.
    if rms.size < 2:
        return None
    signal = float(np.quantile(rms, 0.90))
    if signal <= 0:
        return None  # whole clip is silence
    floor_frames = rms[rms <= np.quantile(rms, 0.05)]
    noise = float(np.median(floor_frames)) if floor_frames.size else float(rms.min())
    if noise <= 0:
        return 120.0  # noise-free floor (digital silence) → clean to the measurement ceiling
    snr = _dbfs(signal) - _dbfs(noise)
    return round(min(snr, 120.0), 2)


def detect_clicks(samples, sr: int, *, max_points: int = 64) -> list[dict]:
    """Click / discontinuity detection via 2nd-difference outliers on the mono mix.

    A click is an isolated sample-to-sample jump far outside the local distribution. Returns
    marker points ({t, sample, label}) at click locations (deduped to one per ~5 ms cluster).
    """
    import numpy as np

    if samples.shape[0] < 3:
        return []
    mono = samples.mean(axis=1).astype(np.float64)
    d2 = np.abs(np.diff(mono, n=2))
    if d2.size == 0:
        return []
    med = float(np.median(d2))
    mad = float(np.median(np.abs(d2 - med))) or 1e-12
    # robust z-score; a true click sits many MADs above the body of the signal
    thresh = med + 12.0 * mad
    # require an absolute jump too, so quiet-but-smooth material doesn't trip
    idx = np.where((d2 > thresh) & (d2 > 0.05))[0]
    if idx.size == 0:
        return []
    points: list[dict] = []
    min_gap = max(1, int(sr * 0.005))  # collapse clusters within 5 ms
    last = -(min_gap + 1)
    for i in idx:
        sample = int(i) + 1  # +1: diff(n=2) index → original sample
        if sample - last < min_gap:
            continue
        last = sample
        points.append({"t": round(sample / sr, 6), "sample": sample, "label": "click"})
        if len(points) >= max_points:
            break
    return points


def detect_gaps(samples, sr: int, *, max_points: int = 64) -> list[dict]:
    """Internal silent-gap / dropout detection.

    Flags runs of >=10 ms of near-digital-silence that are bounded on BOTH sides by signal
    (so leading/trailing silence is not a defect). Returns marker points ({t, sample, dur,
    label}) at each gap onset.
    """
    import numpy as np

    if samples.shape[0] < 3:
        return []
    mono = np.abs(samples).max(axis=1).astype(np.float64)
    silent = mono < 10 ** (-60.0 / 20.0)  # below −60 dBFS
    if not silent.any() or silent.all():
        return []
    min_len = max(1, int(sr * 0.010))  # >=10 ms
    points: list[dict] = []
    n = mono.shape[0]
    i = 0
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        run = j - i
        bounded = i > 0 and j < n  # signal on both sides → an internal dropout
        if run >= min_len and bounded:
            points.append(
                {
                    "t": round(i / sr, 6),
                    "sample": int(i),
                    "dur": round(run / sr, 6),
                    "label": "gap",
                }
            )
            if len(points) >= max_points:
                break
        i = j
    return points


def lossy_origin(samples, sr: int) -> dict[str, Any]:
    """Average-FFT brickwall detector for lossy (MP3/AAC) origin.

    Averages magnitude spectra over the file; walks down from Nyquist to find the highest
    frequency still carrying meaningful energy (the cutoff knee). A hard brickwall well below
    Nyquist (e.g. ~16 kHz for 128 kbps LAME on 44.1 kHz material) flags a likely lossy origin.

    Returns the three registry keys: `qc.lossy.spectral_cutoff_hz`,
    `qc.lossy.expected_nyquist_hz`, `qc.lossy.confidence` (0..1). A high-value FLAG, not proof
    — natural band-limiting and SBR/AAC+ confound it.
    """
    import numpy as np

    nyquist = sr / 2.0
    out = {
        "qc.lossy.spectral_cutoff_hz": round(nyquist, 1),
        "qc.lossy.expected_nyquist_hz": round(nyquist, 1),
        "qc.lossy.confidence": 0.0,
    }
    mono = samples.mean(axis=1).astype(np.float64)
    n = mono.shape[0]
    nfft = 4096
    if n < nfft:
        return out  # too short to resolve a cutoff reliably

    win = np.hanning(nfft)
    hop = nfft // 2
    n_frames = 1 + (n - nfft) // hop
    acc = np.zeros(nfft // 2 + 1, dtype=np.float64)
    for i in range(n_frames):
        seg = mono[i * hop : i * hop + nfft] * win
        acc += np.abs(np.fft.rfft(seg))
    avg = acc / n_frames
    power = avg * avg
    total = float(power.sum())
    if total <= 0:
        return out

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    # cutoff = lowest freq under which `_LOSSY_ENERGY_FRAC` of the energy already lives
    cumfrac = np.cumsum(power) / total
    knee_idx = int(np.searchsorted(cumfrac, _LOSSY_ENERGY_FRAC))
    knee_idx = min(knee_idx, len(freqs) - 1)
    cutoff = float(freqs[knee_idx])
    out["qc.lossy.spectral_cutoff_hz"] = round(cutoff, 1)

    # Knee-below-Nyquist is a GATE, not a linear penalty: full-band content puts its cutoff
    # at Nyquist (knee_frac ~0) and is rejected; anything clearly below Nyquist is eligible,
    # and confidence is then driven by how DEAD the band well above the knee is. (Penalizing
    # a 16 kHz brickwall for being "only" 27% below a 22 kHz Nyquist would wrongly suppress
    # the single most valuable forensic flag.)
    knee_frac = (nyquist - cutoff) / nyquist
    if knee_frac < _LOSSY_MIN_KNEE_FRAC or knee_idx >= len(freqs) - 2:
        return out  # full-band content → not lossy-flagged
    in_band = power[: knee_idx + 1]
    # measure deadness in the upper HALF of the knee→Nyquist span (clear of the knee skirt),
    # where a true brickwall is at the noise floor but natural roll-off still carries energy
    hi_start = knee_idx + 1 + (len(freqs) - 1 - knee_idx) // 2
    above = power[hi_start:]
    in_band_avg = float(in_band.mean()) if in_band.size else 0.0
    above_avg = float(above.mean()) if above.size else 0.0
    if in_band_avg <= 0 or above.size == 0:
        return out
    floor_db = float(10.0 * np.log10((above_avg + 1e-30) / in_band_avg))
    # how far below the in-band level the upper region sits, normalized against the threshold
    confidence = round(min(1.0, max(0.0, (-floor_db) / _LOSSY_FLOOR_DB)), 3)
    out["qc.lossy.confidence"] = float(confidence)
    return out


def qc_audio_frame(audio_frame: dict, *, want_markers: bool = True) -> list[dict]:
    """Run the deterministic QC top-6 over one `audio` frame; return derived frames.

    Emits ONE `feature` frame carrying all the QC scalars (registry `qc.*` keys plus
    `qc.clipping.detected`), plus `marker` frames for click/gap locations.
    Lineage (`of`/`op`/`op_version`/`params`) is set on every derived frame.
    """
    from smplstream import cas, frames as F

    src = cas.get_path(audio_frame["hash"])
    samples, sr = _load(src)
    of = audio_frame.get("id")

    tp = true_peak_dbtp(samples, sr)
    clipping = bool(tp >= _CLIP_DBTP_THRESHOLD)
    corr = phase_correlation(samples)
    dc = dc_offset_dbfs(samples)
    snr = snr_db(samples, sr)
    lossy = lossy_origin(samples, sr)

    feat: dict[str, Any] = {
        # Clipping pass/fail ONLY. true-peak is computed internally to decide it, but the
        # dBTP *measurement* is owned solely by the loudness tier (`loudness.true_peak_dbtp`)
        # — "one measurement, one owner" (feature-keys.md). qc does NOT re-emit it under a
        # qc.* key; run `smpl loudness` for the dBTP number.
        "qc.clipping.detected": clipping,
        "qc.dc_offset_dbfs": round(dc, 2) if dc != float("-inf") else None,
        "qc.snr_db": snr,
        **lossy,
    }
    if corr is not None:
        feat["qc.phase.correlation"] = round(corr, 4)

    params = {"clip_threshold_dbtp": _CLIP_DBTP_THRESHOLD, "sr": sr, "ch": samples.shape[1]}
    out: list[dict] = [
        F.feature_frame(feat, role="qc", of=of, op=OP, op_version=OP_VERSION, params=params)
    ]

    if want_markers:
        clicks = detect_clicks(samples, sr)
        gaps = detect_gaps(samples, sr)
        if clicks:
            out.append(
                F.marker_frame(clicks, role="defect", of=of, op=OP, op_version=OP_VERSION)
            )
        if gaps:
            out.append(
                F.marker_frame(gaps, role="defect", of=of, op=OP, op_version=OP_VERSION)
            )
    return out
