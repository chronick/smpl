"""Tests for the mono-collapse penalty (smpl_analysis.space; ticket vault-1fxy).

The broadband complement to per-band width: how much loudness is lost when the stereo image
sums to mono. Centered/correlated → 0 dB, decorrelated equal-power → ~3 dB, anti-phase →
capped. Mono input → 0 dB. Covers the ranking behavior and the frame contract.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from smpl_analysis import space as SPACE

SR = 44100
KEY = "space.mono_collapse_penalty_db"


def _put_wav(samples, sr=SR):
    from smplstream import cas, frames as F

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    return F.audio_frame(h, sr=meta.get("sr", sr), ch=meta.get("ch", samples.shape[1]),
                         dur=meta.get("dur", samples.shape[0] / sr), role="source")


def _tone(freq=220.0, dur=1.0, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype("float64")


# --- the ranking: mono / correlated / decorrelated / anti-phase ------------------------------


def test_mono_input_zero_penalty():
    d = SPACE.mono_collapse_penalty(_tone(), SR)
    assert d[KEY] == 0.0


def test_correlated_stereo_near_zero():
    x = _tone()
    st = np.column_stack([x, x])
    assert SPACE.mono_collapse_penalty(st, SR)[KEY] < 0.5


def test_decorrelated_stereo_about_3db():
    rng = np.random.default_rng(1)
    n = SR
    left = rng.standard_normal(n) * 0.3
    right = rng.standard_normal(n) * 0.3            # independent, equal power
    st = np.column_stack([left, right])
    p = SPACE.mono_collapse_penalty(st, SR)[KEY]
    assert 2.0 < p < 4.5                             # theory: 10·log10(2) ≈ 3.01 dB


def test_antiphase_is_large_and_ranks_above_decorrelated():
    x = _tone()
    anti = np.column_stack([x, -x])                  # R = −L → mono sum cancels
    rng = np.random.default_rng(2)
    n = SR
    deco = np.column_stack([rng.standard_normal(n) * 0.3, rng.standard_normal(n) * 0.3])
    p_anti = SPACE.mono_collapse_penalty(anti, SR)[KEY]
    p_deco = SPACE.mono_collapse_penalty(deco, SR)[KEY]
    assert p_anti > 20.0
    assert p_anti <= SPACE.CAP_DB
    assert p_anti > p_deco


# --- frame contract --------------------------------------------------------------------------


def test_space_frame_emits_key_and_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    x = _tone()
    af = _put_wav(np.column_stack([x, -x]))          # anti-phase stereo
    derived = SPACE.space_audio_frame(af)
    assert len(derived) == 1
    feat = derived[0]
    assert feat["kind"] == "feature"
    assert feat["role"] == "space"
    assert set(feat["data"].keys()) == {KEY}
    assert feat["data"][KEY] > 20.0                  # collapse detected end-to-end
    assert feat["of"] == af["id"]
    assert feat["op"] == "space"
    assert feat["op_version"] == "space@1"


def test_op_version_constant():
    assert SPACE.OP == "space"
    assert SPACE.OP_VERSION == "space@1"
