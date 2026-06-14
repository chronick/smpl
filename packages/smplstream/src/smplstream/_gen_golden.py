"""Golden-hash corpus: the signals + the regenerator (run after an intentional decode change).

    python -m smplstream._gen_golden

Prints the GOLDEN dict body for tests/test_conformance.py. Changing these is a deliberate
act — it asserts the canonical decode itself moved, which invalidates every cached hash.

The corpus deliberately spans the decode behaviors that DRIFT across libsndfile/numpy
versions — not just near-passthrough FLOAT: integer subtypes (PCM_16/PCM_24 → float scaling),
DOUBLE → float32 narrowing, and multichannel interleave order.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from . import hashing

# (kind, sr, n, subtype)
CASES = [
    ("silence", 8000, 800, "FLOAT"),
    ("dc", 8000, 800, "FLOAT"),
    ("sine", 8000, 800, "FLOAT"),
    ("ramp", 8000, 800, "FLOAT"),
    ("ramp", 8000, 800, "PCM_16"),   # pins int16 → float32 scaling
    ("ramp", 8000, 800, "PCM_24"),   # pins int24 → float32 scaling
    ("third", 8000, 800, "DOUBLE"),  # 1/3 DC: float64 → float32 narrowing (lossy)
    ("stereo_lr", 8000, 800, "FLOAT"),  # pins interleave / channel order (distinct L/R)
]


def signal(kind: str, sr: int, n: int) -> np.ndarray:
    t = np.arange(n) / sr
    if kind == "silence":
        return np.zeros(n, dtype=np.float32)
    if kind == "dc":
        return np.full(n, 0.25, dtype=np.float32)
    if kind == "sine":
        return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    if kind == "ramp":
        return np.linspace(-1.0, 1.0, n, dtype=np.float32)
    if kind == "third":
        # 1/3 has no exact float32 (or float64) representation → narrowing is observable.
        return np.full(n, 1.0 / 3.0, dtype=np.float64)
    if kind == "stereo_lr":
        left = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        right = np.linspace(1.0, -1.0, n, dtype=np.float32)
        return np.stack([left, right], axis=1)
    raise ValueError(kind)


def wav_for_case(kind: str, sr: int, n: int, subtype: str) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, signal(kind, sr, n), sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def main() -> int:
    for kind, sr, n, subtype in CASES:
        h = hashing.audio_hash_bytes(wav_for_case(kind, sr, n, subtype))
        print(f'    ("{kind}", {sr}, {n}, "{subtype}"): "{h}",')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
