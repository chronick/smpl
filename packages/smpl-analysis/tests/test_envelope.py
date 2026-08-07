"""Tests for the aligned-hit envelope (smpl_analysis.envelope; ticket vault-3tuy).

Each test synthesizes a signal with a KNOWN envelope shape (a tight kick, a slow pad, a
two-band loop) and asserts the six pinned scalars land in sane ranges and separate the shapes.
The frame path is exercised through the CAS, asserting registry keys and lineage.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import envelope as ENV

SR = 44100

ENVELOPE_KEYS = {
    "envelope.peak_db_over_floor",
    "envelope.attack_ms_10_90",
    "envelope.rise_slope_db_ms",
    "envelope.t20_ms",
    "envelope.early_decay_slope",
    "envelope.sustain_ratio_150ms",
}


# --- synthesis helpers (SR=44100, no external fixtures) ---------------------------------------


def _noise_floor(n, db=-72.0, seed=0):
    return (10 ** (db / 20.0)) * np.random.default_rng(seed).standard_normal(n)


def tight_kick(dur=0.4, f0=60.0, decay=22.0, sr=SR, lead_silence=0.0):
    """A tight kick: ~1 ms attack from zero, fast exponential decay, short tail."""
    t = np.arange(int(dur * sr)) / sr
    fenv = f0 * np.exp(-t * 8) + f0 * 0.5           # pitch drop
    phase = 2 * np.pi * np.cumsum(fenv) / sr
    amp = (1 - np.exp(-t / 0.001)) * np.exp(-t * decay)  # fast attack, short decay
    body = np.sin(phase) * amp
    body = 0.9 * body / np.max(np.abs(body))
    sil = np.zeros(int(lead_silence * sr))
    sig = np.concatenate([sil, body]) + _noise_floor(len(sil) + len(body), seed=1)
    return sig.astype("float32")


def long_pad(dur=1.5, f0=220.0, attack=0.2, sr=SR, lead_silence=0.0):
    """A long pad: slow (~200 ms) linear attack, then a held sustain."""
    t = np.arange(int(dur * sr)) / sr
    amp = np.clip(t / attack, 0.0, 1.0)
    body = (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)) * amp
    body = 0.7 * body / np.max(np.abs(body))
    sil = np.zeros(int(lead_silence * sr))
    sig = np.concatenate([sil, body]) + _noise_floor(len(sil) + len(body), seed=2)
    return sig.astype("float32")


def two_band_loop(bpm=120, bars=1, sr=SR):
    """A kick on every beat over a sustained sub — the HPSS/sub confound the split resolves."""
    beat = 60.0 / bpm
    n = int(4 * bars * beat * sr)
    t = np.arange(n) / sr
    loop = 0.25 * np.sin(2 * np.pi * 55 * t)        # sustained sub bassline
    for b in range(4 * bars):
        s0 = int(b * beat * sr)
        tt = np.arange(n - s0) / sr
        fenv = 60 * np.exp(-tt * 8) + 40
        ph = 2 * np.pi * np.cumsum(fenv) / sr
        amp = (1 - np.exp(-tt / 0.001)) * np.exp(-tt * 24)
        loop[s0:] += 0.8 * np.sin(ph) * amp         # kick transient every beat
    loop = 0.9 * loop / np.max(np.abs(loop))
    return loop.astype("float32")


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


# --- trim ------------------------------------------------------------------------------------


def test_trim_leading_silence_drops_the_head():
    body = tight_kick(lead_silence=0.0)
    sil = np.zeros(int(0.1 * SR), dtype="float32")
    padded = np.concatenate([sil, body]).astype("float32")
    trimmed, off = ENV.trim_leading_silence(padded, SR)
    assert off >= int(0.05 * SR)          # trimmed at least most of the 100 ms of silence
    assert trimmed.shape[0] < padded.shape[0]


def test_trim_all_silence_is_noop():
    sil = np.zeros(SR, dtype="float32")
    trimmed, off = ENV.trim_leading_silence(sil, SR)
    assert off == 0 and trimmed.shape[0] == sil.shape[0]


# --- scalars: tight kick ---------------------------------------------------------------------


def test_kick_scalars_are_percussive():
    res = ENV.envelope_scalars(tight_kick(), SR, onsets=[0])
    assert res is not None
    d = res["data"]
    assert set(d) == ENVELOPE_KEYS
    assert 0.0 < d["envelope.attack_ms_10_90"] < 30.0        # fast attack
    assert d["envelope.sustain_ratio_150ms"] < 0.3           # little left at 150 ms
    assert d["envelope.early_decay_slope"] < 0.0             # decaying
    assert d["envelope.t20_ms"] is not None and d["envelope.t20_ms"] < 400.0
    assert 3.0 < d["envelope.peak_db_over_floor"] < 130.0


# --- scalars: long pad -----------------------------------------------------------------------


def test_pad_scalars_are_sustained():
    res = ENV.envelope_scalars(long_pad(), SR, onsets=[0])
    assert res is not None
    d = res["data"]
    assert d["envelope.attack_ms_10_90"] > 50.0              # slow attack
    assert d["envelope.sustain_ratio_150ms"] > 0.5           # still going at 150 ms
    # a held pad barely decays early; slope near flat (allow a small negative from noise)
    assert d["envelope.early_decay_slope"] > -0.05
    # never drops 20 dB within the window → t20 is None (censored)
    assert d["envelope.t20_ms"] is None or d["envelope.t20_ms"] > 300.0


def test_kick_and_pad_separate_on_attack_and_sustain():
    k = ENV.envelope_scalars(tight_kick(), SR, onsets=[0])["data"]
    p = ENV.envelope_scalars(long_pad(), SR, onsets=[0])["data"]
    assert k["envelope.attack_ms_10_90"] < p["envelope.attack_ms_10_90"]
    assert k["envelope.sustain_ratio_150ms"] < p["envelope.sustain_ratio_150ms"]


# --- frame emission: one-shot ----------------------------------------------------------------


def test_oneshot_frame_emits_registered_keys_and_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(tight_kick())
    derived = ENV.envelope_audio_frame(af, mode="oneshot")
    feats = [f for f in derived if f["kind"] == "feature"]
    assert len(feats) == 1
    feat = feats[0]
    assert feat["role"] == "envelope"
    assert set(feat["data"].keys()) == ENVELOPE_KEYS
    assert feat["of"] == af["id"]
    assert feat["op"] == "envelope"
    assert feat["op_version"] == "envelope@1"
    assert feat["params"]["mode"] == "oneshot"
    assert feat["params"]["n_hits"] >= 1


# --- frame emission: loop (HPSS + sub split) -------------------------------------------------


def test_loop_frame_splits_percussive_and_sub(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(two_band_loop())
    derived = ENV.envelope_audio_frame(af, mode="loop", bpm=120)
    feats = [f for f in derived if f["kind"] == "feature"]
    roles = {f["role"] for f in feats}
    assert roles == {"envelope.percussive", "envelope.sub"}
    for f in feats:
        assert set(f["data"].keys()) == ENVELOPE_KEYS
        assert f["op_version"] == "envelope@1"
        assert f["of"] == af["id"]
        assert f["params"]["n_hits"] >= 1
    perc = next(f for f in feats if f["role"] == "envelope.percussive")["data"]
    sub = next(f for f in feats if f["role"] == "envelope.sub")["data"]
    # the percussive layer attacks faster and sustains less than the held sub layer
    assert perc["envelope.attack_ms_10_90"] < sub["envelope.attack_ms_10_90"]
    assert perc["envelope.sustain_ratio_150ms"] < sub["envelope.sustain_ratio_150ms"]


def test_auto_mode_picks_loop_for_multi_onset(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(two_band_loop())
    derived = ENV.envelope_audio_frame(af, mode="auto")
    roles = {f["role"] for f in derived if f["kind"] == "feature"}
    assert roles == {"envelope.percussive", "envelope.sub"}


def test_auto_mode_picks_oneshot_for_single_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(tight_kick())
    derived = ENV.envelope_audio_frame(af, mode="auto")
    roles = {f["role"] for f in derived if f["kind"] == "feature"}
    assert roles == {"envelope"}


def test_op_version_constant():
    assert ENV.OP == "envelope"
    assert ENV.OP_VERSION == "envelope@1"
