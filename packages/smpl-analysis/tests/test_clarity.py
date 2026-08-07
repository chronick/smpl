"""Tests for the clarity feature family (smpl_analysis.clarity; ticket vault-1fxy).

The load-bearing case (acceptance): a muddy pad (energy piled in the 200–500 Hz low-mids) and
a bright stab (energy in the 2–6 kHz presence band) must RANK correctly on mud_presence_ratio —
the pad reads mud-heavy (>1), the stab presence-heavy (<1). Corroborating clarity signals
(low_mid_masking, presence_focus, presence_transient) rank the same way. Also covers the
duration gate and the frame contract.

Signals are synthesized at SR=44100.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from smpl_analysis import clarity as CL

SR = 44100
KEYS = set(CL.KEYS)


# --- synthesis helpers -----------------------------------------------------------------------


def _fade(x, sr, fade_in_ms=0.0, fade_out_ms=0.0):
    """Raised-cosine fades so an abrupt buffer edge doesn't inject spectral splatter."""
    x = x.copy()
    ni = int(fade_in_ms / 1000.0 * sr)
    no = int(fade_out_ms / 1000.0 * sr)
    if ni:
        x[:ni] *= 0.5 * (1 - np.cos(np.linspace(0, np.pi, ni)))
    if no:
        x[-no:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, no)))
    return x


def muddy_pad(dur=1.5, sr=SR):
    """A sustained pad whose energy sits entirely in the 200–500 Hz low-mid (mud) region.

    Soft attack/release (like a real pad) so it has no hard buffer edges — the presence band
    is genuinely SUSTAINED (flat per-frame energy → low transient crest).
    """
    t = np.arange(int(dur * sr)) / sr
    body = (0.5 * np.sin(2 * np.pi * 250 * t)
            + 0.4 * np.sin(2 * np.pi * 350 * t)
            + 0.3 * np.sin(2 * np.pi * 450 * t))
    body = _fade(body, sr, fade_in_ms=80.0, fade_out_ms=80.0)
    return (0.8 * body / np.max(np.abs(body))).astype("float32")


def bright_stab(dur=0.8, sr=SR):
    """A short, transient stab whose energy sits in the 2–6 kHz presence band (fast decay).

    Sharp onset (the transient we measure) + exponential decay to silence.
    """
    t = np.arange(int(dur * sr)) / sr
    tone = (np.sin(2 * np.pi * 3000 * t)
            + np.sin(2 * np.pi * 4000 * t)
            + np.sin(2 * np.pi * 5000 * t))
    env = np.exp(-t * 25.0)                       # sharp attack, fast decay → transient
    body = _fade(tone * env, sr, fade_out_ms=20.0)  # tail already ~0; keep the onset sharp
    return (0.8 * body / np.max(np.abs(body))).astype("float32")


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


# --- the acceptance ranking: muddy pad vs bright stab ----------------------------------------


def test_muddy_vs_bright_ranks_on_mud_presence():
    m = CL.clarity_scalars(muddy_pad(), SR)
    b = CL.clarity_scalars(bright_stab(), SR)
    assert set(m) == KEYS and set(b) == KEYS

    # headline: the pad is mud-heavy (>1), the stab presence-heavy (<1), pad ≫ stab.
    assert m["clarity.mud_presence_ratio"] > b["clarity.mud_presence_ratio"]
    assert m["clarity.mud_presence_ratio"] > 1.0
    assert b["clarity.mud_presence_ratio"] < 1.0


def test_corroborating_clarity_signals_rank_the_same_way():
    m = CL.clarity_scalars(muddy_pad(), SR)
    b = CL.clarity_scalars(bright_stab(), SR)
    # low-mid masking: the pad's 200–500 buildup dominates the core mids; the stab does not.
    assert m["clarity.low_mid_masking_db"] > b["clarity.low_mid_masking_db"]
    # presence focus: the stab concentrates energy in the 2–6 kHz presence band.
    assert b["clarity.presence_focus_ratio"] > m["clarity.presence_focus_ratio"]
    # presence transient: the stab's presence is percussive (high crest); the pad's is sustained.
    assert b["clarity.presence_transient_ratio"] > m["clarity.presence_transient_ratio"]


# --- duration gate ---------------------------------------------------------------------------


def test_short_material_is_gated_to_null(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(muddy_pad(dur=0.2))             # below MIN_DURATION_S (0.5 s)
    feat = CL.clarity_audio_frame(af)[0]
    assert feat["params"]["gated"] is True
    assert set(feat["data"]) == KEYS
    assert all(v is None for v in feat["data"].values())


def test_stab_above_gate_still_emits(tmp_path, monkeypatch):
    # a 0.8 s stab is above the 0.5 s clarity gate → real values, not nulls.
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(bright_stab(dur=0.8))
    feat = CL.clarity_audio_frame(af)[0]
    assert feat["params"]["gated"] is False
    assert all(v is not None for v in feat["data"].values())


# --- frame contract --------------------------------------------------------------------------


def test_clarity_frame_emits_keys_and_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(muddy_pad())
    derived = CL.clarity_audio_frame(af)
    assert len(derived) == 1
    feat = derived[0]
    assert feat["kind"] == "feature"
    assert feat["role"] == "clarity"
    assert set(feat["data"].keys()) == KEYS
    assert feat["of"] == af["id"]
    assert feat["op"] == "clarity"
    assert feat["op_version"] == "clarity@1"


def test_clarity_reuses_the_standardized_band_edges():
    # clarity must not redefine band edges — it reuses width's registered six.
    from smpl_analysis import width as W

    edges = CL._band_edges()
    assert set(edges) == {b for b, _lo, _hi in W.BANDS if b != "full"}
    assert edges["lomid"] == (200.0, 500.0)
    assert edges["uppermid"] == (2000.0, 6000.0)


def test_op_version_constant():
    assert CL.OP == "clarity"
    assert CL.OP_VERSION == "clarity@1"
