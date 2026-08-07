"""Tests for the movement feature family (smpl_analysis.movement; ticket vault-1fxy).

The load-bearing case (acceptance): a pumping / sidechained loop and a static drone must
RANK correctly — the pumping loop swings wide on sidechain_db and hf_mod_depth_db while the
drone barely moves. Also covers the LUFS-style duration gate (short material → all keys null)
and the frame contract (registry keys, role, lineage).

Signals are synthesized at SR=44100 with no external fixtures.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from smpl_analysis import movement as MV

SR = 44100
KEYS = set(MV.KEYS)


# --- synthesis helpers -----------------------------------------------------------------------


def _hp(x, fc, sr=SR):
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, fc / (sr / 2), btype="high", output="sos")
    return sosfiltfilt(sos, x)


def pumping_loop(dur=2.0, bpm=120, sr=SR, seed=5):
    """A sustained low pad + bright top layer, both DUCKED once per beat (a sidechain pump)."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    beat = 60.0 / bpm
    carrier = 0.5 * np.sin(2 * np.pi * 110 * t) + 0.3 * np.sin(2 * np.pi * 220 * t)
    hf = _hp(rng.standard_normal(n), 6500) * 0.25          # bright top layer (>6 kHz)
    duck = 0.08 + 0.92 * ((t % beat) / beat)               # ducked at each beat, ramps back up
    sig = (carrier + hf) * duck
    return (0.9 * sig / np.max(np.abs(sig))).astype("float32")


def static_drone(dur=2.0, sr=SR, seed=5):
    """The SAME pad + top layer, held constant — no modulation of any kind."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    carrier = 0.5 * np.sin(2 * np.pi * 110 * t) + 0.3 * np.sin(2 * np.pi * 220 * t)
    hf = _hp(rng.standard_normal(n), 6500) * 0.25          # steady top layer, NOT ducked
    sig = carrier + hf
    return (0.9 * sig / np.max(np.abs(sig))).astype("float32")


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


# --- the acceptance ranking: pumping loop vs static drone ------------------------------------


def test_pumping_vs_static_ranks_on_sidechain_and_hf_mod():
    p = MV.movement_scalars(pumping_loop(), SR)
    s = MV.movement_scalars(static_drone(), SR)
    assert set(p) == KEYS and set(s) == KEYS

    # pump depth: the ducked loop swings wide; the drone is nearly flat.
    assert p["movement.sidechain_db"] > s["movement.sidechain_db"]
    assert p["movement.sidechain_db"] > 6.0
    assert s["movement.sidechain_db"] < 3.0

    # HF modulation: the ducked top layer swings; the steady top layer does not.
    assert p["movement.hf_mod_depth_db"] > s["movement.hf_mod_depth_db"]
    assert p["movement.hf_mod_depth_db"] > 6.0
    assert s["movement.hf_mod_depth_db"] < 3.0


def test_static_drone_has_low_hf_silence():
    # a continuous top layer is never HF-silent → ~0 %.
    s = MV.movement_scalars(static_drone(), SR)
    assert s["movement.hf_silence_pct"] < 5.0


def test_bass_only_drone_reads_as_hf_silent():
    # a pure low sub with no top end → HF sits far below the full-band peak → ~100 % silent.
    t = np.arange(int(2.0 * SR)) / SR
    sub = (0.7 * np.sin(2 * np.pi * 55 * t)).astype("float32")
    d = MV.movement_scalars(sub, SR)
    assert d["movement.hf_silence_pct"] > 90.0


# --- duration gate ---------------------------------------------------------------------------


def test_short_material_is_gated_to_null(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(pumping_loop(dur=0.3))          # below MIN_DURATION_S (1.0 s)
    feat = MV.movement_audio_frame(af)[0]
    assert feat["params"]["gated"] is True
    assert set(feat["data"]) == KEYS
    assert all(v is None for v in feat["data"].values())


def test_long_material_is_not_gated(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(pumping_loop(dur=2.0))
    feat = MV.movement_audio_frame(af)[0]
    assert feat["params"]["gated"] is False
    assert all(v is not None for v in feat["data"].values())


# --- frame contract --------------------------------------------------------------------------


def test_movement_frame_emits_keys_and_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(pumping_loop())
    derived = MV.movement_audio_frame(af)
    assert len(derived) == 1
    feat = derived[0]
    assert feat["kind"] == "feature"
    assert feat["role"] == "movement"
    assert set(feat["data"].keys()) == KEYS
    assert feat["of"] == af["id"]
    assert feat["op"] == "movement"
    assert feat["op_version"] == "movement@1"


def test_op_version_constant():
    assert MV.OP == "movement"
    assert MV.OP_VERSION == "movement@1"
