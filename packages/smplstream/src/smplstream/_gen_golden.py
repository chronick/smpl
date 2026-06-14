"""Regenerate the golden-hash corpus constants (run after an intentional decoder change).

    python -m smplstream._gen_golden

Prints the GOLDEN dict body for tests/test_conformance.py. Changing these is a deliberate
act — it asserts the canonical decode itself moved, which invalidates every cached hash.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from . import hashing

CASES = [("silence", 8000, 800, "FLOAT"), ("dc", 8000, 800, "FLOAT"),
         ("sine", 8000, 800, "FLOAT"), ("ramp", 8000, 800, "FLOAT")]


def _signal(kind: str, sr: int, n: int) -> np.ndarray:
    t = np.arange(n) / sr
    return {
        "silence": np.zeros(n, dtype=np.float32),
        "dc": np.full(n, 0.25, dtype=np.float32),
        "sine": (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32),
        "ramp": np.linspace(-1.0, 1.0, n, dtype=np.float32),
    }[kind]


def main() -> int:
    for kind, sr, n, subtype in CASES:
        buf = io.BytesIO()
        sf.write(buf, _signal(kind, sr, n), sr, format="WAV", subtype=subtype)
        h = hashing.audio_hash_bytes(buf.getvalue())
        print(f'    ("{kind}", {sr}, {n}, "{subtype}"): "{h}",')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
