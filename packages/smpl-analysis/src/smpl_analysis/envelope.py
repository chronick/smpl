"""Aligned-hit amplitude envelope (council DSP recipe; ticket vault-3tuy).

The "how does this hit move in time" axis — the shape of the attack and decay averaged
across the hits in a sample, robust to onset jitter. The recipe is **pinned** so independent
implementations agree:

  1. trim leading silence (below −60 dBFS)
  2. onset detect — spectral-flux onset strength, ``backtrack``ed to the local minimum,
     minimum spacing = half a 16th note at a known BPM (falls back to a fixed floor spacing
     when no BPM is supplied)
  3. per-onset windows — ``[onset − 10 ms, onset + min(IOI, 500 ms)]``
  4. ALIGN each window on its 10 %-of-peak amplitude crossing — onset timestamps jitter
     ±2–5 ms, and averaging *unaligned* windows lowpasses the attack (smears the very thing
     we want to measure)
  5. |hilbert| analytic envelope through an asymmetric follower (~5 ms attack / ~20 ms
     release), converted to dB
  6. per-time-bin **median** across the aligned hits (median, not mean — one bad hit can't
     drag the shape)

From the median envelope, six pinned scalars:

  ``peak_db_over_floor``   — peak level above the quietest-sustained floor (5th-percentile of
                             the envelope; dB). Large for a one-shot that returns to silence,
                             small for a pad that never gets quiet — i.e. a dynamic-range read.
  ``attack_ms_10_90``      — 10 %→90 %-of-peak rise time (ms)
  ``rise_slope_db_ms``     — mean rise slope over that attack (dB/ms)
  ``t20_ms``               — time to decay 20 dB below the peak (ms; None if it never does)
  ``early_decay_slope``    — least-squares decay slope over the first 50 ms after the peak (dB/ms)
  ``sustain_ratio_150ms``  — linear level 150 ms after the crossing, as a fraction of the peak

**Loops** (more than one onset): the recipe runs TWICE, on the HPSS-percussive component
AND on a sub band (LPF 120 Hz), separately — otherwise a kick tail and a bassline note that
land together confound each other's envelope. Onsets are detected on each component in its
own band, so the two envelopes describe two different layers. Two `feature` frames come out,
roles ``envelope.percussive`` and ``envelope.sub``.

**One-shots** (one onset or forced): a single-hit passthrough — the trimmed clip is one hit,
reported under role ``envelope``.

**Pre-limiter material only.** A brickwall limiter flattens the very dynamics these scalars
measure (attack, decay, sustain-ratio), so a limited loop lies about them. This op does not
detect limiting; the caller is responsible for feeding it pre-limiter audio.

Pure functions returning frame dicts. Heavy imports (librosa/scipy/numpy) live inside the
functions so cold pipe stages start fast.
"""

from __future__ import annotations

from typing import Optional

OP = "envelope"
OP_VERSION = "envelope@1"

# --- pinned recipe constants (declared so memo params are complete & stable per op_version) --
SILENCE_DB = -60.0        # leading-silence trim threshold (dBFS)
HOP_LENGTH = 512          # onset-strength STFT hop
PRE_MS = 10.0             # window pre-roll before the onset / alignment point (ms)
POST_MS = 500.0           # window post-roll cap = min(IOI, POST_MS) (ms)
ATTACK_MS = 5.0           # envelope-follower attack time constant (ms)
RELEASE_MS = 20.0         # envelope-follower release time constant (ms)
CROSS_FRAC = 0.10         # alignment crossing = this fraction of the hit peak (the "10 %")
EARLY_DECAY_MS = 50.0     # window for the early-decay least-squares slope (ms)
SUSTAIN_MS = 150.0        # sustain-ratio probe time after the crossing (ms)
MIN_SPACING_S = 0.030     # onset min-spacing floor when no BPM is known (s)
SUB_LPF_HZ = 120.0        # sub-band low-pass cutoff for the loop split (Hz)

_EPS = 1e-12
_FLOOR_LIN = 1e-6         # hard floor (≈ −120 dBFS) so a silent pre-roll can't yield −inf dB


# --------------------------------------------------------------------------------------------
# Low-level DSP helpers (pure numpy/scipy).
# --------------------------------------------------------------------------------------------


def _to_mono(y):
    import numpy as np

    y = np.asarray(y, dtype="float64")
    if y.ndim > 1:
        # accept (ch, n) or (n, ch); collapse the channel axis
        y = y.mean(axis=0) if y.shape[0] < y.shape[1] else y.mean(axis=1)
    return np.ascontiguousarray(y)


def trim_leading_silence(y, sr: int, *, thresh_db: float = SILENCE_DB) -> tuple:
    """Drop everything before the first sample that rises above ``thresh_db`` dBFS.

    Returns ``(trimmed, offset_samples)``. An all-silent signal is returned unchanged with
    offset 0 (there is nothing to align to).
    """
    import numpy as np

    amp = np.abs(y)
    thr = 10.0 ** (thresh_db / 20.0)
    above = amp > thr
    if not above.any():
        return y, 0
    start = int(np.argmax(above))
    return y[start:], start


def _butter_lowpass(y, sr: int, cutoff_hz: float, order: int = 4):
    """Zero-phase Butterworth low-pass (matches the sosfiltfilt style used across smpl-analysis)."""
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    wn = min(max(cutoff_hz / nyq, 1e-4), 0.999)
    sos = butter(order, wn, btype="low", output="sos")
    # sosfiltfilt needs more samples than its pad length; guard tiny signals.
    if y.shape[0] <= 3 * order * 2:
        return np.asarray(y, dtype="float64")
    return sosfiltfilt(sos, y)


def follower_envelope(y, sr: int, *, attack_ms: float = ATTACK_MS,
                      release_ms: float = RELEASE_MS):
    """|hilbert| analytic amplitude through an asymmetric one-pole follower.

    Fast attack tracks the transient; slower release smooths the decay so the envelope reads
    cleanly without lowpassing the onset. Returns a linear-amplitude envelope the length of
    the input.
    """
    import numpy as np
    from scipy.signal import hilbert

    y = np.asarray(y, dtype="float64")
    if y.size == 0:
        return y
    amp = np.abs(hilbert(y))
    a_att = float(np.exp(-1.0 / max(sr * attack_ms / 1000.0, 1.0)))
    a_rel = float(np.exp(-1.0 / max(sr * release_ms / 1000.0, 1.0)))
    env = np.empty_like(amp)
    prev = amp[0]
    for i, x in enumerate(amp):
        coef = a_att if x > prev else a_rel
        prev = coef * prev + (1.0 - coef) * x
        env[i] = prev
    return env


def _onset_samples(y, sr: int, *, bpm: Optional[float] = None, hop: int = HOP_LENGTH):
    """Spectral-flux onsets, backtracked, with BPM-aware minimum spacing.

    ``min spacing = half a 16th note`` at ``bpm`` (``(60/bpm)/8`` s); with no BPM we fall back
    to a fixed ``MIN_SPACING_S`` floor. Returns onset positions in sample indices.
    """
    import librosa
    import numpy as np

    y = np.asarray(y, dtype="float32")
    if y.size < hop * 2:
        return np.array([0], dtype=int)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)  # spectral flux
    min_spacing = (60.0 / bpm) / 8.0 if bpm else MIN_SPACING_S
    wait = max(1, int(round(min_spacing * sr / hop)))
    frames = librosa.onset.onset_detect(
        onset_envelope=oenv, sr=sr, hop_length=hop, backtrack=True, units="frames", wait=wait
    )
    if frames is None or len(frames) == 0:
        return np.array([0], dtype=int)
    return librosa.frames_to_samples(np.asarray(frames), hop_length=hop).astype(int)


def _aligned_median(env, sr: int, onsets, *, pre_ms: float = PRE_MS,
                    post_ms: float = POST_MS, cross_frac: float = CROSS_FRAC):
    """Window each onset, align on the ``cross_frac``-of-peak rising crossing, median across hits.

    Returns ``(median_env_linear, pre_samples, n_hits)``. ``median_env_linear`` is on a common
    grid whose index ``pre_samples`` is the alignment point (t = 0). Internal NaNs (hits shorter
    than the grid) are interpolated so the returned grid is fully finite. Returns
    ``(None, pre_samples, 0)`` when no usable hit remains.
    """
    import warnings

    import numpy as np

    pre = max(1, int(round(pre_ms / 1000.0 * sr)))
    post = max(2, int(round(post_ms / 1000.0 * sr)))
    grid_len = pre + post
    n = env.shape[0]
    onsets = [int(o) for o in onsets]

    hits: list = []
    for i, on in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else n
        end = min(on + post, nxt, n)          # window end = min(IOI, POST_MS)
        start = max(0, on - pre)              # window start = onset − PRE_MS
        w = env[start:end]
        if w.size < 3:
            continue
        peak = float(np.max(w))
        if peak <= _EPS:
            continue
        thr = cross_frac * peak
        crossed = np.nonzero(w >= thr)[0]
        if crossed.size == 0:
            continue
        cross = int(crossed[0])              # first rising crossing of cross_frac·peak
        # Place `cross` at grid index `pre`: aligned[j] = w[cross - pre + j].
        idxs = np.arange(grid_len) + (cross - pre)
        aligned = np.full(grid_len, np.nan)
        valid = (idxs >= 0) & (idxs < w.size)
        aligned[valid] = w[idxs[valid]]
        hits.append(aligned)

    if not hits:
        return None, pre, 0

    stack = np.vstack(hits)
    with warnings.catch_warnings():  # an all-NaN column (grid edges) → NaN, filled below
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(stack, axis=0)

    good = np.isfinite(med)
    if good.sum() < 2:
        return None, pre, len(hits)
    # Interpolate internal NaNs; np.interp clamps to edge values outside the finite span.
    gx = np.nonzero(good)[0]
    med = np.interp(np.arange(grid_len), gx, med[gx])
    return med, pre, len(hits)


def _scalars_from_median(med, sr: int, pre: int, *, floor_lin: Optional[float] = None) -> dict:
    """The six pinned scalars from a finite linear median envelope (t=0 at index ``pre``).

    ``floor_lin`` is the noise-floor estimate (linear amplitude) the peak is measured above;
    when omitted it falls back to a low percentile of the median envelope.
    """
    import numpy as np

    lin = np.asarray(med, dtype="float64")
    db = 20.0 * np.log10(np.maximum(lin, _EPS))

    peak_idx = int(np.argmax(lin))
    peak_lin = float(lin[peak_idx])
    peak_db = 20.0 * np.log10(max(peak_lin, _EPS))

    # --- floor: a global noise-floor estimate (passed in) or a low pct of this envelope. ---
    if floor_lin is None:
        floor_lin = float(np.percentile(lin, 5))
    floor_lin = max(floor_lin, _FLOOR_LIN)
    peak_db_over_floor = peak_db - 20.0 * np.log10(floor_lin)

    # --- attack 10 %→90 %: grid index `pre` IS the 10 % crossing (t=0). ---
    align_idx = pre
    seg = lin[align_idx:peak_idx + 1]
    if seg.size:
        hit90 = np.nonzero(seg >= 0.9 * peak_lin)[0]
        idx90 = align_idx + int(hit90[0]) if hit90.size else peak_idx
    else:
        idx90 = peak_idx
    attack_samps = max(idx90 - align_idx, 0)
    attack_ms = attack_samps / sr * 1000.0

    # --- rise slope over the attack (dB/ms). ---
    if attack_ms > 0:
        rise_slope = (db[idx90] - db[align_idx]) / attack_ms
    else:
        rise_slope = None  # instantaneous attack (single-bin rise) — slope is undefined

    # --- t20: peak → 20 dB below peak (ms); None if it never gets there in-window. ---
    after = db[peak_idx:]
    below = np.nonzero(after <= peak_db - 20.0)[0]
    t20_ms = float(below[0]) / sr * 1000.0 if below.size else None

    # --- early decay: least-squares slope of dB vs ms over EARLY_DECAY_MS after the peak. ---
    early_n = max(2, int(round(EARLY_DECAY_MS / 1000.0 * sr)))
    early_db = db[peak_idx:peak_idx + early_n]
    if early_db.size >= 2:
        t_ms = np.arange(early_db.size) / sr * 1000.0
        early_decay_slope = float(np.polyfit(t_ms, early_db, 1)[0])
    else:
        early_decay_slope = None

    # --- sustain ratio: linear level SUSTAIN_MS after the crossing / peak. ---
    idx150 = align_idx + int(round(SUSTAIN_MS / 1000.0 * sr))
    idx150 = min(idx150, lin.size - 1)
    sustain_ratio = float(lin[idx150] / peak_lin) if peak_lin > _EPS else 0.0

    # Cast every value to a native Python float (numpy 2.x scalar reprs like
    # "np.float64(…)" break the frame content-id canonicalizer, cf. spectral.py).
    return {
        "envelope.peak_db_over_floor": round(float(peak_db_over_floor), 3),
        "envelope.attack_ms_10_90": round(float(attack_ms), 3),
        "envelope.rise_slope_db_ms": round(float(rise_slope), 4) if rise_slope is not None else None,
        "envelope.t20_ms": round(float(t20_ms), 3) if t20_ms is not None else None,
        "envelope.early_decay_slope": round(float(early_decay_slope), 4) if early_decay_slope is not None else None,
        "envelope.sustain_ratio_150ms": round(float(sustain_ratio), 4),
    }


# --------------------------------------------------------------------------------------------
# Component pipeline: signal → aligned median → scalars.
# --------------------------------------------------------------------------------------------


def envelope_scalars(y, sr: int, *, bpm: Optional[float] = None,
                     onsets=None) -> Optional[dict]:
    """Run the full aligned-hit pipeline on one (already band-limited) mono signal.

    Returns ``{"data": <six-scalars>, "n_hits": int}`` or ``None`` when no usable hit exists.
    ``onsets`` may be supplied (sample indices) to skip detection (one-shot passthrough).
    """
    import numpy as np

    y = _to_mono(y)
    if y.size < 8:
        return None
    if onsets is None:
        onsets = _onset_samples(y, sr, bpm=bpm)
    onsets = np.asarray(sorted(int(o) for o in onsets), dtype=int)
    if onsets.size == 0:
        onsets = np.array([0], dtype=int)

    env = follower_envelope(y, sr)
    # Noise floor: the quietest sustained level of the whole follower envelope (robust for
    # both one-shots — the decayed tail — and loops — the inter-onset gaps).
    floor_lin = float(np.percentile(env, 5)) if env.size else _FLOOR_LIN
    med, pre, n_hits = _aligned_median(env, sr, onsets)
    if med is None:
        return None
    data = _scalars_from_median(med, sr, pre, floor_lin=floor_lin)
    return {"data": data, "n_hits": n_hits}


# --------------------------------------------------------------------------------------------
# Frame-emitting entrypoint (the op behind `smpl envelope` and describe-all).
# --------------------------------------------------------------------------------------------


def envelope_audio_frame(
    audio_frame: dict,
    *,
    mode: str = "auto",
    bpm: Optional[float] = None,
) -> list[dict]:
    """Resolve an `audio` frame's PCM and emit its aligned-hit envelope feature frame(s).

    ``mode``:
      - ``"oneshot"`` — single-hit passthrough → one `feature` frame, role ``envelope``.
      - ``"loop"``    — HPSS-percussive + sub-band (LPF 120 Hz) split → two `feature` frames,
                        roles ``envelope.percussive`` / ``envelope.sub``.
      - ``"auto"``    — one-shot if ≤1 onset is detected, else loop.

    Each derived frame carries ``of``/``op``/``op_version``/``params`` lineage; ``params``
    records the mode, band/component, hit count, BPM, and the pinned window constants. The
    caller is responsible for passthrough of the input frame.
    """
    import numpy as np
    import soundfile as sf

    from smplstream import cas, error_frame, frames as F

    of = audio_frame.get("id")
    src = cas.get_path(audio_frame["hash"])
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)
    y = y.T  # (ch, n)
    mono = _to_mono(y)
    trimmed, _off = trim_leading_silence(mono, sr)

    base_params = {
        "silence_db": SILENCE_DB, "hop_length": HOP_LENGTH, "pre_ms": PRE_MS,
        "post_ms": POST_MS, "attack_ms": ATTACK_MS, "release_ms": RELEASE_MS,
        "cross_frac": CROSS_FRAC, "early_decay_ms": EARLY_DECAY_MS, "sustain_ms": SUSTAIN_MS,
        "sub_lpf_hz": SUB_LPF_HZ,
    }

    resolved = mode
    if mode == "auto":
        onsets = _onset_samples(trimmed, sr, bpm=bpm)
        resolved = "loop" if len(onsets) >= 2 else "oneshot"

    out: list[dict] = []

    def _emit(signal, *, role: str, band: str, onsets=None):
        res = envelope_scalars(signal, sr, bpm=bpm, onsets=onsets)
        if res is None:
            out.append(error_frame(
                "op_failed", f"envelope: no usable hit in band {band}", of=of, op=OP
            ))
            return
        params = {**base_params, "mode": resolved, "band": band,
                  "n_hits": res["n_hits"], "bpm": bpm}
        out.append(F.feature_frame(
            res["data"], role=role, of=of, op=OP, op_version=OP_VERSION, params=params
        ))

    if resolved == "oneshot":
        # single-hit passthrough: treat the trimmed clip as one hit anchored at its start
        _emit(trimmed, role="envelope", band="fullband", onsets=[0])
    else:
        import librosa

        # HPSS-percussive component (kick/transient layer)
        try:
            perc = librosa.effects.hpss(np.asarray(trimmed, dtype="float32"))[1]
        except Exception:
            perc = trimmed
        _emit(perc, role="envelope.percussive", band="percussive")
        # sub band (LPF 120 Hz) — the bassline/kick-body layer
        sub = _butter_lowpass(trimmed, sr, SUB_LPF_HZ)
        _emit(sub, role="envelope.sub", band="sub")

    return out
