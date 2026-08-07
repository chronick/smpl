"""Mono-collapse penalty — the broadband complement to per-band width (ticket vault-1fxy).

The per-band width family (vault-3tuy) answers *which band* mono-collapses; this answers
*how much loudness is lost overall* when the stereo image is summed to mono — the single
number a broadcast/club-mono check cares about.

``mono_collapse_penalty_db`` = ``10·log10(mean-per-channel-power / mono-sum-power)``, where
the mono sum is ``(L+R)/2``:

  - identical L/R (centered, correlated) → mono sum keeps the level → **0 dB** (no penalty)
  - decorrelated equal-power L/R          → mono sum loses half the power → **~3 dB**
  - anti-phase (R = −L)                   → mono sum cancels → penalty caps at :data:`CAP_DB`

Mono input (1 channel) is trivially mono-compatible → **0 dB**. NOT duration-gated (a
static stereo relationship is well-defined on any length, including one-shots — same as the
width family).

Layering rule: emits the measurement only; "how much collapse is acceptable" lives in
profiles / docs. Pure functions; heavy imports stay inside the functions.
"""

from __future__ import annotations

OP = "space"
OP_VERSION = "space@1"

CAP_DB = 60.0  # penalty ceiling for (near-)total cancellation, so a dead mono sum is finite
_EPS = 1e-12

KEYS: tuple[str, ...] = ("space.mono_collapse_penalty_db",)


def mono_collapse_penalty(samples, sr: int) -> dict:
    """Mono-collapse penalty (dB ≥ 0) for a ``(n, ch)`` (or 1-D mono) float array.

    Returns ``{"space.mono_collapse_penalty_db": <float>}``. Mono input → 0.0.
    """
    import numpy as np

    s = np.asarray(samples, dtype="float64")
    if s.ndim == 1:
        s = s[:, None]

    if s.shape[1] < 2:  # mono → trivially mono-compatible, no penalty
        return {"space.mono_collapse_penalty_db": 0.0}

    left = s[:, 0]
    right = s[:, 1]
    ref_power = 0.5 * (float(np.mean(left * left)) + float(np.mean(right * right)))
    mono = 0.5 * (left + right)
    mono_power = float(np.mean(mono * mono))

    if ref_power <= _EPS:
        penalty = 0.0                        # both channels silent → nothing to collapse
    elif mono_power <= _EPS:
        penalty = CAP_DB                     # total cancellation → capped penalty
    else:
        penalty = 10.0 * np.log10(ref_power / mono_power)
    penalty = max(0.0, min(float(penalty), CAP_DB))
    return {"space.mono_collapse_penalty_db": round(penalty, 3)}


def space_audio_frame(audio_frame: dict) -> list[dict]:
    """Resolve an `audio` frame's PCM and emit its mono-collapse feature frame (role ``space``).

    Returns a one-element list with the derived `feature` frame carrying ``of``/``op``/
    ``op_version``/``params`` lineage. The caller passes the input frame through.
    """
    import soundfile as sf

    from smplstream import cas, frames as F

    src = cas.get_path(audio_frame["hash"])
    samples, sr = sf.read(str(src), dtype="float32", always_2d=True)  # (n, ch)

    data = mono_collapse_penalty(samples, sr)
    params = {"cap_db": CAP_DB, "ch": int(samples.shape[1]), "sr": int(sr)}
    return [
        F.feature_frame(
            data, role="space", of=audio_frame["id"], op=OP, op_version=OP_VERSION,
            params=params,
        )
    ]
