"""Shared fixtures: an isolated CAS per test so the real ~/.smpl is never touched."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture()
def isolated_cas(tmp_path, monkeypatch):
    cas_dir = tmp_path / "cas"
    monkeypatch.setenv("SMPL_CAS_DIR", str(cas_dir))
    return cas_dir


@pytest.fixture()
def tone_wav_bytes():
    def _make(freq=440.0, sr=48000, dur=0.25, ch=1, subtype="FLOAT"):
        t = np.arange(int(sr * dur)) / sr
        mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        samples = mono if ch == 1 else np.stack([mono] * ch, axis=1)
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV", subtype=subtype)
        return buf.getvalue()

    return _make
