"""Edit operations — the `smpl filter / eq / env / fx / slice` DSP tier (ticket vault-3l83).

Pure functions over a resolved audio frame: each loads the canonical PCM from the CAS,
applies a transform, re-CASes the result, and returns new smplstream frame dicts. The thin
CLI subcommands in the `smpl` package call into these.

Two families:
  - **Audio-producing ops** (`apply_filter`, `apply_eq`, `apply_env`, `apply_fx`) return a
    new `audio` frame with role ``<role>.wet`` (the dry→wet convention) and full lineage
    (``of`` / ``lineage`` / ``op`` / ``op_version`` / ``params``). Filtering/EQ/envelope use
    scipy (pure-Python, deterministic, empty env-fingerprint); fx (reverb/delay) shells out
    to ``sox`` and fingerprints the tool version.
  - **Marker-producing op** (`slice_onsets`) runs librosa onset detection and returns a
    ``marker`` frame (role ``onset``) plus, optionally, one sliced ``audio`` frame per
    region (role ``slice:<n>``).

Heavy imports (librosa, scipy, soundfile, matplotlib) stay INSIDE the functions so a cold
pipe stage starts fast. No new dependencies — scipy/librosa/soundfile/numpy are installed,
sox/ffmpeg are on PATH.
"""

from __future__ import annotations

import io
import math
import subprocess
from typing import Optional

# ---------------------------------------------------------------------------
# op_version constants — bumped on ANY behavior change (spec → *Memoization*).
# ---------------------------------------------------------------------------
FILTER_OP_VERSION = "filter@1"
EQ_OP_VERSION = "eq@1"
ENV_OP_VERSION = "env@1"
FX_OP_VERSION = "fx@1"
SLICE_OP_VERSION = "slice@1"
GAIN_OP_VERSION = "gain@1"
NORMALIZE_OP_VERSION = "normalize@1"
LIMIT_OP_VERSION = "limit@1"


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------
def _load_audio(audio_frame: dict):
    """Resolve an audio frame's CAS blob to ``(samples (frames, ch) float32, sr)``."""
    import soundfile as sf

    from smplstream import cas

    src = cas.get_path(audio_frame["hash"])
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    return data, int(sr)


def _wet_role(audio_frame: dict) -> str:
    """Derive the ``<role>.wet`` role from the source frame's role (default ``edit``)."""
    role = audio_frame.get("role") or "edit"
    # Strip an existing .wet/.dry suffix so re-filtering stays ``<base>.wet`` (not .wet.wet).
    for suffix in (".wet", ".dry"):
        if role.endswith(suffix):
            role = role[: -len(suffix)]
            break
    return f"{role}.wet"


def _emit_wet_audio(
    samples,
    sr: int,
    *,
    src_frame: dict,
    op: str,
    op_version: str,
    params: dict,
) -> dict:
    """CAS a processed (frames, ch) float32 array as WAV and return a wet `audio` frame."""
    import numpy as np
    import soundfile as sf

    from smplstream import cas, frames as F

    arr = np.ascontiguousarray(np.asarray(samples, dtype="float32"))
    if arr.ndim == 1:
        arr = arr[:, None]
    # WAV back-patches its RIFF size header → needs a seekable sink; render to memory.
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    return F.audio_frame(
        h,
        sr=meta.get("sr", sr),
        ch=meta.get("ch", arr.shape[1]),
        dur=meta.get("dur", arr.shape[0] / sr if sr else 0.0),
        role=_wet_role(src_frame),
        of=src_frame.get("id"),
        lineage=[src_frame["id"]] if src_frame.get("id") else None,
        op=op,
        op_version=op_version,
        params=params,
        fmt=meta.get("fmt"),
    )


def _sox_version_fingerprint() -> str:
    from smplstream import memo

    return memo.tool_version_fingerprint(["sox", "--version"])


# ---------------------------------------------------------------------------
# filter — high/low/band-pass via scipy Butterworth (deterministic, pure-Python).
# ---------------------------------------------------------------------------
def apply_filter(
    audio_frame: dict,
    *,
    kind: str,
    freq,
    order: int = 4,
) -> dict:
    """Apply an HP/LP/BP Butterworth filter, returning a wet `audio` frame.

    ``kind`` ∈ {"hp", "lp", "bp"}. For ``bp``, ``freq`` is a ``(low_hz, high_hz)`` pair;
    otherwise a single cutoff in Hz. Zero-phase (``filtfilt``) so no group-delay smear.
    """
    import numpy as np
    from scipy.signal import butter, filtfilt, sosfiltfilt

    data, sr = _load_audio(audio_frame)
    nyq = sr / 2.0

    if kind == "bp":
        low, high = float(freq[0]), float(freq[1])
        wn = [max(low / nyq, 1e-6), min(high / nyq, 0.999999)]
        sos = butter(order, wn, btype="bandpass", output="sos")
        out = sosfiltfilt(sos, data, axis=0)
        params = {"kind": "bp", "low_hz": low, "high_hz": high, "order": order, "sr_hz": sr}
    else:
        cutoff = float(freq)
        btype = {"hp": "highpass", "lp": "lowpass"}[kind]
        wn = min(max(cutoff / nyq, 1e-6), 0.999999)
        b, a = butter(order, wn, btype=btype)
        out = filtfilt(b, a, data, axis=0)
        params = {"kind": kind, "freq_hz": cutoff, "order": order, "sr_hz": sr}

    out = np.clip(out, -1.0, 1.0).astype("float32")
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="filter", op_version=FILTER_OP_VERSION, params=params
    )


# ---------------------------------------------------------------------------
# eq — peaking / shelving bands via scipy biquad (RBJ cookbook coefficients).
# ---------------------------------------------------------------------------
def _biquad_peaking(f0: float, q: float, gain_db: float, sr: int):
    import numpy as np

    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = 1 + alpha * a_amp
    b1 = -2 * cos_w0
    b2 = 1 - alpha * a_amp
    a0 = 1 + alpha / a_amp
    a1 = -2 * cos_w0
    a2 = 1 - alpha / a_amp
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _biquad_shelf(f0: float, gain_db: float, sr: int, *, high: bool, slope: float = 1.0):
    import numpy as np

    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((a_amp + 1 / a_amp) * (1 / slope - 1) + 2)
    two_sqrt_a_alpha = 2 * np.sqrt(a_amp) * alpha
    if high:
        b0 = a_amp * ((a_amp + 1) + (a_amp - 1) * cos_w0 + two_sqrt_a_alpha)
        b1 = -2 * a_amp * ((a_amp - 1) + (a_amp + 1) * cos_w0)
        b2 = a_amp * ((a_amp + 1) + (a_amp - 1) * cos_w0 - two_sqrt_a_alpha)
        a0 = (a_amp + 1) - (a_amp - 1) * cos_w0 + two_sqrt_a_alpha
        a1 = 2 * ((a_amp - 1) - (a_amp + 1) * cos_w0)
        a2 = (a_amp + 1) - (a_amp - 1) * cos_w0 - two_sqrt_a_alpha
    else:
        b0 = a_amp * ((a_amp + 1) - (a_amp - 1) * cos_w0 + two_sqrt_a_alpha)
        b1 = 2 * a_amp * ((a_amp - 1) - (a_amp + 1) * cos_w0)
        b2 = a_amp * ((a_amp + 1) - (a_amp - 1) * cos_w0 - two_sqrt_a_alpha)
        a0 = (a_amp + 1) + (a_amp - 1) * cos_w0 + two_sqrt_a_alpha
        a1 = -2 * ((a_amp - 1) + (a_amp + 1) * cos_w0)
        a2 = (a_amp + 1) + (a_amp - 1) * cos_w0 - two_sqrt_a_alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def apply_eq(audio_frame: dict, *, bands: list[dict]) -> dict:
    """Apply a chain of EQ bands, returning a wet `audio` frame.

    Each band: ``{"type": "peaking"|"lowshelf"|"highshelf", "freq": Hz, "gain": dB,
    "q": float}`` (``q`` used by peaking; shelves use a unit slope).
    """
    import numpy as np
    from scipy.signal import lfilter

    data, sr = _load_audio(audio_frame)
    out = data.astype("float64", copy=True)
    norm_bands = []
    for band in bands:
        btype = band.get("type", "peaking")
        f0 = float(band["freq"])
        gain_db = float(band.get("gain", 0.0))
        q = float(band.get("q", 1.0))
        if btype == "peaking":
            b, a = _biquad_peaking(f0, q, gain_db, sr)
        elif btype == "lowshelf":
            b, a = _biquad_shelf(f0, gain_db, sr, high=False)
        elif btype == "highshelf":
            b, a = _biquad_shelf(f0, gain_db, sr, high=True)
        else:
            raise ValueError(f"unknown eq band type: {btype!r}")
        out = lfilter(b, a, out, axis=0)
        norm_bands.append({"type": btype, "freq_hz": f0, "gain_db": gain_db, "q": q})

    out = np.clip(out, -1.0, 1.0).astype("float32")
    params = {"bands": norm_bands, "sr_hz": sr}
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="eq", op_version=EQ_OP_VERSION, params=params
    )


# ---------------------------------------------------------------------------
# env — amplitude envelope (pluck / fade / gate), deterministic numpy.
# ---------------------------------------------------------------------------
def apply_env(
    audio_frame: dict,
    *,
    shape: str,
    attack: float = 0.0,
    release: float = 0.0,
    threshold_db: float = -40.0,
) -> dict:
    """Apply an amplitude envelope, returning a wet `audio` frame.

    ``shape`` ∈ {"pluck", "fade", "gate"}:
      - ``pluck`` — fast linear attack (``attack`` s) then exponential decay over ``release`` s.
      - ``fade``  — linear fade-in (``attack`` s) and fade-out (``release`` s).
      - ``gate``  — silence samples whose short-term level is below ``threshold_db`` (dBFS).
    """
    import numpy as np

    data, sr = _load_audio(audio_frame)
    n = data.shape[0]
    t = np.arange(n) / sr if sr else np.arange(n)

    if shape == "pluck":
        gain = np.ones(n, dtype="float64")
        a = max(int(attack * sr), 1)
        gain[:a] = np.linspace(0.0, 1.0, a)
        if release > 0:
            tau = release / 5.0  # ~5 time-constants to ≈0 over the release window
            decay = np.exp(-(t - t[a - 1]) / tau)
            decay[:a] = 1.0
            gain = gain * decay
        params = {"shape": "pluck", "attack_s": attack, "release_s": release, "sr_hz": sr}
    elif shape == "fade":
        gain = np.ones(n, dtype="float64")
        a = min(max(int(attack * sr), 0), n)
        r = min(max(int(release * sr), 0), n)
        if a > 0:
            gain[:a] = np.linspace(0.0, 1.0, a)
        if r > 0:
            gain[n - r:] = np.linspace(1.0, 0.0, r)
        params = {"shape": "fade", "attack_s": attack, "release_s": release, "sr_hz": sr}
    elif shape == "gate":
        mono = data.mean(axis=1)
        win = max(int(0.01 * sr), 1)  # 10 ms RMS window
        kernel = np.ones(win) / win
        env = np.sqrt(np.convolve(mono ** 2, kernel, mode="same") + 1e-12)
        thresh_lin = 10.0 ** (threshold_db / 20.0)
        gain = (env >= thresh_lin).astype("float64")
        params = {"shape": "gate", "threshold_db": threshold_db, "sr_hz": sr}
    else:
        raise ValueError(f"unknown env shape: {shape!r}")

    out = (data * gain[:, None]).astype("float32")
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="env", op_version=ENV_OP_VERSION, params=params
    )


# ---------------------------------------------------------------------------
# fx — reverb / delay via sox (shell-out; env-fingerprinted tool version).
# ---------------------------------------------------------------------------
def apply_fx(
    audio_frame: dict,
    *,
    effect: str,
    amount: float = 50.0,
    delay_ms: float = 250.0,
    decay: float = 0.5,
) -> dict:
    """Apply a sox-driven effect, returning a wet `audio` frame.

    ``effect`` ∈ {"reverb", "delay"}. ``reverb`` uses sox ``reverb <amount>`` (0–100).
    ``delay`` uses sox ``echo`` with one tap at ``delay_ms`` / ``decay``.
    """
    from smplstream import cas

    src = cas.get_path(audio_frame["hash"])
    if effect == "reverb":
        chain = ["reverb", str(float(amount))]
        params = {"effect": "reverb", "amount": float(amount)}
    elif effect == "delay":
        # sox echo: gain-in gain-out <delay_ms decay> ...
        chain = ["echo", "0.8", "0.9", str(float(delay_ms)), str(float(decay))]
        params = {"effect": "delay", "delay_ms": float(delay_ms), "decay": float(decay)}
    else:
        raise ValueError(f"unknown fx effect: {effect!r}")

    # Render float32 WAV to stdout so we never silently truncate bit depth.
    cmd = ["sox", str(src), "-t", "wav", "-e", "floating-point", "-b", "32", "-", *chain]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"sox {effect} failed: {proc.stderr.decode('utf-8', 'replace').strip()}")

    h = cas.put_audio_bytes(proc.stdout)
    meta = cas.read_meta(h) or {}

    from smplstream import frames as F

    params["env_fingerprint"] = _sox_version_fingerprint()
    return F.audio_frame(
        h,
        sr=meta.get("sr", 0),
        ch=meta.get("ch", 1),
        dur=meta.get("dur", 0.0),
        role=_wet_role(audio_frame),
        of=audio_frame.get("id"),
        lineage=[audio_frame["id"]] if audio_frame.get("id") else None,
        op="fx",
        op_version=FX_OP_VERSION,
        params=params,
        fmt=meta.get("fmt"),
    )


# ---------------------------------------------------------------------------
# gain / normalize / limit — level management (ticket vault-3l83 follow-up).
#
# The level-management trio. All deterministic numpy (empty env-fingerprint), all emit a
# wet ``<role>.wet`` audio frame carrying the measured-before / applied-gain in ``params`` so
# the level decision is auditable from the lineage. They compose like every other op — the
# canonical leveling chain is ``read | normalize --lufs -14 | limit | write``:
#
#   - ``gain``      — the primitive: a pure dB scale, NO clipping (float-safe, composable;
#                     pair with ``limit`` for a ceiling). The single-knob building block.
#   - ``normalize`` — LUFS-normalize to a target (BS.1770 integrated), WITH a true-peak
#                     ceiling so it is safe standalone AND composes. Trades exact-LUFS for
#                     no-clipping when the target would breach the ceiling.
#   - ``limit``     — a true-peak ceiling by whole-sample gain reduction (never boosts). Not
#                     a look-ahead compressor: transparent, no pumping, preserves dynamics —
#                     the right primitive for one-shot / sample prep.
# ---------------------------------------------------------------------------
def apply_gain(audio_frame: dict, *, db: float) -> dict:
    """Scale the selected audio frame by ``db`` decibels, returning a wet `audio` frame.

    A pure level change — deliberately **not** clipped, so it is lossless and composes (a
    boost that exceeds 0 dBFS survives in the float CAS blob; follow with ``limit`` for a
    delivery ceiling). The single-knob primitive ``normalize`` is built on.
    """
    import numpy as np

    data, sr = _load_audio(audio_frame)
    factor = 10.0 ** (float(db) / 20.0)
    out = (data.astype("float64") * factor).astype("float32")
    params = {"db": float(db), "sr_hz": sr}
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="gain", op_version=GAIN_OP_VERSION, params=params
    )


def apply_normalize(
    audio_frame: dict,
    *,
    target_lufs: float,
    ceiling_dbtp: Optional[float] = -1.0,
) -> dict:
    """Loudness-normalize to ``target_lufs`` (BS.1770 integrated), returning a wet frame.

    Computes the gain that moves measured integrated LUFS to ``target_lufs``. If applying
    that gain would push the true peak above ``ceiling_dbtp`` (default −1 dBTP), the gain is
    pulled back so the true peak lands exactly at the ceiling — i.e. the op self-limits,
    trading a little loudness for no inter-sample clipping. Pass ``ceiling_dbtp=None`` to
    normalize to exact LUFS with no ceiling (then chain ``limit`` yourself). Silent / too-
    short signals pass through ungained with a ``note`` in params.

    The measured-in loudness, true peak, and applied gain are recorded in ``params`` so the
    level decision is fully auditable from the frame lineage.
    """
    import numpy as np

    from . import loudness

    data, sr = _load_audio(audio_frame)
    res = loudness.analyze_array(data, sr)
    measured_lufs = res["integrated_lufs"]
    measured_tp = res["true_peak_dbtp"]

    note = None
    if measured_lufs is None or not math.isfinite(measured_lufs):
        gain_db = 0.0
        note = "silent_or_too_short: passed through ungained"
    else:
        gain_db = float(target_lufs) - float(measured_lufs)

    ceiling_applied = False
    if (
        ceiling_dbtp is not None
        and measured_tp is not None
        and math.isfinite(measured_tp)
    ):
        projected_tp = measured_tp + gain_db
        if projected_tp > ceiling_dbtp:
            gain_db -= projected_tp - ceiling_dbtp
            ceiling_applied = True

    factor = 10.0 ** (gain_db / 20.0)
    out = (data.astype("float64") * factor).astype("float32")
    params = {
        "target_lufs": float(target_lufs),
        "ceiling_dbtp": ceiling_dbtp,
        "measured_lufs_in": loudness._db_round(measured_lufs),
        "measured_true_peak_dbtp_in": loudness._db_round(measured_tp),
        "applied_gain_db": round(float(gain_db), 3),
        "ceiling_applied": ceiling_applied,
        "sr_hz": sr,
    }
    if note:
        params["note"] = note
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="normalize",
        op_version=NORMALIZE_OP_VERSION, params=params,
    )


def apply_limit(audio_frame: dict, *, ceiling_dbtp: float = -1.0) -> dict:
    """Guarantee true peak ≤ ``ceiling_dbtp`` by whole-sample gain reduction (never boosts).

    Measures the 4×-oversampled true peak and, if it exceeds the ceiling, scales the entire
    sample down so the worst inter-sample peak sits exactly at the ceiling. A transparent,
    deterministic ceiling — not a look-ahead compressor — so it never pumps and preserves
    the sample's dynamics. Below-ceiling input passes through unchanged (gain 0). The right
    safety stage for one-shots and for the tail of a normalize chain.
    """
    import numpy as np

    from . import loudness

    data, sr = _load_audio(audio_frame)
    res = loudness.analyze_array(data, sr)
    measured_tp = res["true_peak_dbtp"]

    gain_db = 0.0
    if measured_tp is not None and math.isfinite(measured_tp) and measured_tp > ceiling_dbtp:
        gain_db = float(ceiling_dbtp) - float(measured_tp)  # always ≤ 0

    factor = 10.0 ** (gain_db / 20.0)
    out = (data.astype("float64") * factor).astype("float32")
    params = {
        "ceiling_dbtp": float(ceiling_dbtp),
        "measured_true_peak_dbtp_in": loudness._db_round(measured_tp),
        "applied_gain_db": round(float(gain_db), 3),
        "sr_hz": sr,
    }
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="limit", op_version=LIMIT_OP_VERSION, params=params
    )


# ---------------------------------------------------------------------------
# slice — librosa onset detection → marker frame (+ optional sliced audio frames).
# ---------------------------------------------------------------------------
def slice_onsets(
    audio_frame: dict,
    *,
    emit_audio: bool = False,
    backtrack: bool = True,
) -> list[dict]:
    """Detect onsets and return a `marker` frame; optionally one sliced `audio` per region.

    The marker frame (role ``onset``) carries one point per onset with float-second ``t`` and
    sample-accurate ``sample`` (spec → *Units & timebase*: markers destined for sample-exact
    export MUST carry ``sample``). When ``emit_audio`` is set, each inter-onset region is also
    CASed and emitted as an ``audio`` frame with role ``slice:<n>``.
    """
    import librosa
    import numpy as np

    from smplstream import frames as F

    data, sr = _load_audio(audio_frame)
    mono = data.mean(axis=1) if data.ndim > 1 else data

    onset_samples = librosa.onset.onset_detect(
        y=mono, sr=sr, backtrack=backtrack, units="samples"
    )
    onset_samples = [int(s) for s in onset_samples]

    points = [
        {"t": round(s / sr, 6), "sample": s, "label": f"onset-{i}"}
        for i, s in enumerate(onset_samples)
    ]
    out: list[dict] = [
        F.marker_frame(
            points,
            role="onset",
            of=audio_frame.get("id"),
            op="slice",
            op_version=SLICE_OP_VERSION,
            lineage=[audio_frame["id"]] if audio_frame.get("id") else None,
            params={"backtrack": backtrack, "emit_audio": emit_audio, "sr_hz": sr},
        )
    ]

    if emit_audio and onset_samples:
        import io as _io

        import soundfile as sf

        from smplstream import cas

        bounds = onset_samples + [data.shape[0]]
        for i in range(len(onset_samples)):
            start, end = bounds[i], bounds[i + 1]
            if end <= start:
                continue
            region = np.ascontiguousarray(data[start:end], dtype="float32")
            buf = _io.BytesIO()
            sf.write(buf, region, sr, format="WAV", subtype="FLOAT")
            h = cas.put_audio_bytes(buf.getvalue())
            meta = cas.read_meta(h) or {}
            out.append(
                F.audio_frame(
                    h,
                    sr=meta.get("sr", sr),
                    ch=meta.get("ch", region.shape[1] if region.ndim > 1 else 1),
                    dur=meta.get("dur", (end - start) / sr if sr else 0.0),
                    role=f"slice:{i}",
                    of=audio_frame.get("id"),
                    lineage=[audio_frame["id"]] if audio_frame.get("id") else None,
                    op="slice",
                    op_version=SLICE_OP_VERSION,
                    params={"index": i, "start_sample": start, "end_sample": end, "sr_hz": sr},
                    fmt=meta.get("fmt"),
                )
            )
    return out
