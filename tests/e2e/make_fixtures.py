"""Generate a small, deterministic audio fixture corpus for e2e tests.

Real-library e2e runs point SMPL_E2E_SAMPLES at an actual sample dir (e.g. the mini's
library); this corpus is the always-available fallback so the suite runs anywhere. Signals
are chosen to exercise distinct analysis paths: tonal, percussive, stereo, near-clipping,
and a band-limited (lossy-looking) case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _write(path: Path, data: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float32), sr, subtype=subtype)


def main(out_dir: str) -> int:
    out = Path(out_dir)
    sr = 44100

    # 1. Tonal sustained note (A3) — tonal/spectral paths.
    t = np.arange(int(sr * 1.5)) / sr
    _write(out / "tone_a3.wav", 0.5 * np.sin(2 * np.pi * 220.0 * t), sr)

    # 2. Percussive: exponentially-decaying noise bursts at 120 BPM — onset/rhythm paths.
    n = int(sr * 2.0)
    sig = np.zeros(n, dtype=np.float32)
    period = int(sr * 0.5)
    rng = np.random.default_rng(0)
    for start in range(0, n, period):
        env = np.exp(-30.0 * np.arange(min(period, n - start)) / sr)
        sig[start:start + len(env)] += (rng.standard_normal(len(env)) * env).astype(np.float32)
    _write(out / "perc_120bpm.wav", sig * 0.6, sr)

    # 3. Stereo with channel difference — phase/mono-compat paths.
    left = 0.4 * np.sin(2 * np.pi * 330.0 * t)
    right = 0.4 * np.sin(2 * np.pi * 330.0 * t + 0.6)
    _write(out / "stereo_pad.wav", np.stack([left, right], axis=1), sr)

    # 4. Near-clipping loud signal — clipping/true-peak/loudness paths.
    hot = 0.99 * np.sin(2 * np.pi * 110.0 * t)
    _write(out / "hot_bass.wav", hot, sr, subtype="PCM_24")

    # 5. Band-limited to ~11 kHz (mimics lossy origin) — lossy-cutoff QC path.
    full = rng.standard_normal(int(sr * 1.0)).astype(np.float32) * 0.2
    spec = np.fft.rfft(full)
    freqs = np.fft.rfftfreq(len(full), 1 / sr)
    spec[freqs > 11000] = 0.0
    _write(out / "lossy_like.wav", np.fft.irfft(spec, n=len(full)).astype(np.float32), sr)

    print(f"wrote {len(list(out.glob('*.wav')))} fixtures to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "tests/e2e/_work/fixtures"))
