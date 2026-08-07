"""Tests for smpl_analysis.octave — 1/6-octave spectrum feature (ticket vault-22oy)."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from smplstream import cas, frames as F

from smpl_analysis import octave as OCT

SR = 44100


def _tone(freq, sr=SR, dur=0.5, amp=0.4):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


@pytest.fixture()
def cas_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    return tmp_path


def _audio_frame(sig, sr=SR):
    if sig.ndim == 1:
        sig = sig.reshape(-1, 1)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    return F.audio_frame(h, sr=meta.get("sr", sr), ch=meta.get("ch", 1),
                         dur=meta.get("dur", sig.shape[0] / sr), role="source")


def test_octave_centers_are_ascending_and_finite():
    centers, db = OCT.octave_spectrum(_tone(220.0), SR)
    assert centers.size == db.size and centers.size > 10
    assert np.all(np.diff(centers) > 0)           # strictly ascending
    assert np.all(np.isfinite(db))
    assert np.all(db <= 0.0 + 1e-6)               # normalized to total in-band power → ≤ 0 dB


def test_low_tone_puts_energy_in_low_bands():
    """A 60 Hz tone should read hotter in sub/bass than in air (shape, not level)."""
    bands = OCT.band_levels(_tone(60.0), SR)
    assert bands["bass"] > bands["air"]
    assert bands["sub"] > bands["mid"]


def test_scalars_carry_octave_and_band_keys():
    data = OCT.octave_spectrum_scalars(_tone(440.0), SR)
    assert any(k.startswith("spectrum.oct6.") for k in data)
    for name in ("sub", "bass", "lomid", "mid", "uppermid", "air"):
        assert f"spectrum.band.{name}" in data


def test_octave_audio_frame_shape_and_lineage(cas_dir):
    af = _audio_frame(_tone(330.0))
    out = OCT.octave_audio_frame(af)
    assert len(out) == 1
    f = out[0]
    assert f["kind"] == "feature"
    assert f["role"] == "octave-spectrum"
    assert f["op_version"] == "octave-spectrum@1"
    assert f["of"] == af["id"]
    assert F.validate_frame(f) == []
    assert any(k.startswith("spectrum.oct6.") for k in f["data"])
