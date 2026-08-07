"""Tests for per-band stereo width / correlation (smpl_analysis.width; ticket vault-3tuy).

The load-bearing case: a texture that is WIDE in the highs but MONO in the sub. A single
broadband number looks moderately wide and hides the fact that the sub is mono-safe — the
per-band split must show sub correlation ≈ 1 / width ≈ 0 while the air band is decorrelated.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import width as W

SR = 44100

ALL_KEYS = {f"width.{b}.{m}" for b, _lo, _hi in W.BANDS
            for m in ("correlation", "side_mid_ratio")}


def _hp(x, fc, sr=SR):
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, fc / (sr / 2), btype="high", output="sos")
    return sosfiltfilt(sos, x)


def wide_highs_mono_sub(dur=1.0, sr=SR, seed=3):
    """Mono sub (identical L/R) + independently-generated (decorrelated) high bands."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    sub = 0.5 * np.sin(2 * np.pi * 40 * t)              # mono sub, same in both channels
    hi_l = _hp(rng.standard_normal(n), 4000) * 0.3
    hi_r = _hp(rng.standard_normal(n), 4000) * 0.3      # independent → decorrelated highs
    left = (sub + hi_l).astype("float32")
    right = (sub + hi_r).astype("float32")
    return np.column_stack([left, right])


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


# --- mono ------------------------------------------------------------------------------------


def test_mono_is_fully_correlated_zero_width():
    mono = (0.5 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)).astype("float32")
    d = W.per_band_width(mono, SR)
    assert set(d.keys()) == ALL_KEYS
    for b, _lo, _hi in W.BANDS:
        assert d[f"width.{b}.correlation"] == 1.0
        assert d[f"width.{b}.side_mid_ratio"] == 0.0


# --- the wide-sub-hides-in-broadband case ----------------------------------------------------


def test_wide_highs_mono_sub_separates_per_band():
    d = W.per_band_width(wide_highs_mono_sub(), SR)
    # sub is mono → high correlation, ~zero width
    assert d["width.sub.correlation"] > 0.9
    assert d["width.sub.side_mid_ratio"] < 0.1
    # air is decorrelated → low correlation, substantial width
    assert d["width.air.correlation"] < 0.5
    assert d["width.air.side_mid_ratio"] > 0.5
    # the sub is clearly narrower than the air band (the thing broadband would hide)
    assert d["width.sub.side_mid_ratio"] < d["width.air.side_mid_ratio"]


def test_broadband_masks_the_mono_sub():
    d = W.per_band_width(wide_highs_mono_sub(), SR)
    # a single full-band read sits between the bands — it neither exposes the wide air nor
    # the mono sub, which is exactly why the per-band split exists.
    assert d["width.full.side_mid_ratio"] > d["width.sub.side_mid_ratio"]
    assert d["width.full.side_mid_ratio"] < d["width.air.side_mid_ratio"]


# --- frame emission --------------------------------------------------------------------------


def test_width_frame_emits_keys_and_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(wide_highs_mono_sub())
    derived = W.width_audio_frame(af)
    assert len(derived) == 1
    feat = derived[0]
    assert feat["kind"] == "feature"
    assert feat["role"] == "width"
    assert set(feat["data"].keys()) == ALL_KEYS
    assert feat["of"] == af["id"]
    assert feat["op"] == "width"
    assert feat["op_version"] == "width@1"
    # the six standardized band edges are recorded in params
    assert set(feat["params"]["bands"]) == {b for b, _lo, _hi in W.BANDS}


def test_band_edges_are_the_six_standardized_bands():
    names = {b for b, _lo, _hi in W.BANDS}
    assert {"sub", "bass", "lomid", "mid", "uppermid", "air", "full"} == names
    edges = {b: (lo, hi) for b, lo, hi in W.BANDS}
    assert edges["sub"] == (20.0, 60.0)
    assert edges["bass"] == (60.0, 200.0)
    assert edges["lomid"] == (200.0, 500.0)
    assert edges["mid"] == (500.0, 2000.0)
    assert edges["uppermid"] == (2000.0, 6000.0)
    assert edges["air"] == (6000.0, 20000.0)


def test_op_version_constant():
    assert W.OP == "width"
    assert W.OP_VERSION == "width@1"
