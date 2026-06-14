"""Tests for smpl_analysis.edit — the edit filters (ticket vault-3l83).

Covers filter (HP/LP/BP), eq, env, fx (sox), slice (onset markers + sliced audio), and the
`smpl select` stream filter. Each audio-producing op is checked for: a wet `audio` frame with
role ``<role>.wet``, correct lineage (``of``/``lineage``/``op``/``op_version``/``params``),
and the actual DSP effect on the signal. slice/select are asserted at minimum per the brief.
"""

from __future__ import annotations

import io
import shutil

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import edit


SR = 44100


def _tone(freq=440.0, dur=1.0, amp=0.5, sr=SR, channels=1):
    t = np.arange(int(dur * sr)) / sr
    sig = amp * np.sin(2 * np.pi * freq * t)
    if channels == 1:
        return sig.reshape(-1, 1).astype(np.float32)
    return np.column_stack([sig] * channels).astype(np.float32)


def _two_tones(low=120.0, high=8000.0, dur=1.0, amp=0.4, sr=SR):
    """A signal with energy at a low and a high frequency, for filter assertions."""
    t = np.arange(int(dur * sr)) / sr
    sig = amp * (np.sin(2 * np.pi * low * t) + np.sin(2 * np.pi * high * t))
    return sig.reshape(-1, 1).astype(np.float32)


def _put_wav(samples, sr, role="source"):
    from smplstream import cas, frames as F

    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    return F.audio_frame(
        h,
        sr=meta.get("sr", sr),
        ch=meta.get("ch", samples.shape[1]),
        dur=meta.get("dur", samples.shape[0] / sr),
        role=role,
    )


def _load_frame_samples(frame):
    from smplstream import cas

    data, sr = sf.read(str(cas.get_path(frame["hash"])), dtype="float32", always_2d=True)
    return data, sr


def _band_energy(samples, sr, lo, hi):
    """RMS energy in a frequency band via FFT (for filter effect checks)."""
    mono = samples.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sqrt(np.mean(spec[mask] ** 2))) if mask.any() else 0.0


def _assert_wet_lineage(frame, src, op, op_version):
    assert frame["kind"] == "audio"
    assert frame["role"] == "source.wet"
    assert frame.get("of") == src["id"]
    assert frame.get("lineage") == [src["id"]]
    assert frame.get("op") == op
    assert frame.get("op_version") == op_version
    assert isinstance(frame.get("params"), dict)
    # a new content hash (the op changed the bytes)
    assert frame["hash"] != src["hash"]


# --- filter ----------------------------------------------------------------------------


def test_filter_highpass_attenuates_low_band(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_two_tones(), SR)
    wet = edit.apply_filter(src, kind="hp", freq=1000.0)
    _assert_wet_lineage(wet, src, "filter", edit.FILTER_OP_VERSION)
    assert wet["params"]["kind"] == "hp" and wet["params"]["freq_hz"] == 1000.0

    before, sr = _load_frame_samples(src)
    after, _ = _load_frame_samples(wet)
    # the 120 Hz energy should drop sharply; the 8 kHz energy should survive
    assert _band_energy(after, sr, 50, 300) < 0.25 * _band_energy(before, sr, 50, 300)
    assert _band_energy(after, sr, 7000, 9000) > 0.5 * _band_energy(before, sr, 7000, 9000)


def test_filter_lowpass_attenuates_high_band(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_two_tones(), SR)
    wet = edit.apply_filter(src, kind="lp", freq=1000.0)
    before, sr = _load_frame_samples(src)
    after, _ = _load_frame_samples(wet)
    assert _band_energy(after, sr, 7000, 9000) < 0.25 * _band_energy(before, sr, 7000, 9000)
    assert _band_energy(after, sr, 50, 300) > 0.5 * _band_energy(before, sr, 50, 300)


def test_filter_bandpass(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_two_tones(low=120.0, high=8000.0), SR)
    wet = edit.apply_filter(src, kind="bp", freq=(2000.0, 5000.0))
    assert wet["params"]["kind"] == "bp"
    after, sr = _load_frame_samples(wet)
    before, _ = _load_frame_samples(src)
    # both the 120 Hz and 8 kHz tones (outside the 2-5k band) should be attenuated
    assert _band_energy(after, sr, 50, 300) < 0.3 * _band_energy(before, sr, 50, 300)
    assert _band_energy(after, sr, 7000, 9000) < 0.3 * _band_energy(before, sr, 7000, 9000)


# --- eq --------------------------------------------------------------------------------


def test_eq_peaking_boost_raises_band(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_two_tones(low=200.0, high=4000.0), SR)
    wet = edit.apply_eq(src, bands=[{"type": "peaking", "freq": 4000.0, "gain": 12.0, "q": 2.0}])
    _assert_wet_lineage(wet, src, "eq", edit.EQ_OP_VERSION)
    before, sr = _load_frame_samples(src)
    after, _ = _load_frame_samples(wet)
    assert _band_energy(after, sr, 3500, 4500) > 1.5 * _band_energy(before, sr, 3500, 4500)


# --- env -------------------------------------------------------------------------------


def test_env_fade_zeroes_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_tone(amp=0.6), SR)
    wet = edit.apply_env(src, shape="fade", attack=0.05, release=0.05)
    _assert_wet_lineage(wet, src, "env", edit.ENV_OP_VERSION)
    after, _ = _load_frame_samples(wet)
    assert abs(after[0, 0]) < 1e-3 and abs(after[-1, 0]) < 1e-3
    # midpoint region stays loud (single-sample check would hit a sine zero-crossing)
    mid = len(after) // 2
    assert np.max(np.abs(after[mid - 500:mid + 500, 0])) > 0.1


def test_env_gate_silences_quiet_region(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    s = _tone(amp=0.6, dur=0.5)
    quiet = _tone(amp=0.0005, dur=0.5)
    sig = np.vstack([s, quiet]).astype(np.float32)
    src = _put_wav(sig, SR)
    wet = edit.apply_env(src, shape="gate", threshold_db=-40.0)
    after, _ = _load_frame_samples(wet)
    n = len(s)
    assert np.max(np.abs(after[n + 1000:])) < 1e-3  # quiet tail gated out
    assert np.max(np.abs(after[:n])) > 0.1          # loud head survives


# --- fx (sox) --------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("sox") is None, reason="sox not on PATH")
def test_fx_reverb_changes_signal_and_fingerprints(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_tone(amp=0.4, dur=0.5), SR)
    wet = edit.apply_fx(src, effect="reverb", amount=60.0)
    _assert_wet_lineage(wet, src, "fx", edit.FX_OP_VERSION)
    assert wet["params"]["effect"] == "reverb"
    assert wet["params"]["amount"] == 60.0
    # env fingerprint of the shell-out tool is recorded for memoization
    assert wet["params"].get("env_fingerprint")


# --- slice -----------------------------------------------------------------------------


def _click_train(n_clicks=4, dur=2.0, sr=SR):
    """Silence with sharp impulses → clean, well-separated onsets."""
    sig = np.zeros((int(dur * sr), 1), dtype=np.float32)
    spacing = len(sig) // (n_clicks + 1)
    for i in range(1, n_clicks + 1):
        pos = i * spacing
        # a short decaying burst so librosa's spectral flux fires
        burst = np.hanning(200) * 0.8
        sig[pos:pos + 200, 0] = burst.astype(np.float32)
    return sig


def test_slice_emits_onset_markers(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_click_train(), SR)
    out = edit.slice_onsets(src, emit_audio=False)
    markers = [f for f in out if f["kind"] == "marker"]
    assert len(markers) == 1
    m = markers[0]
    assert m["role"] == "onset"
    assert m.get("of") == src["id"]
    assert m.get("op") == "slice" and m.get("op_version") == edit.SLICE_OP_VERSION
    assert len(m["data"]) >= 3, "expected onsets for the click train"
    for p in m["data"]:
        assert "t" in p and "sample" in p
        assert isinstance(p["sample"], int)
        # sample/t consistency against native sr (spec → Units & timebase)
        assert abs(p["sample"] / SR - p["t"]) < 1e-3
    # no audio frames when emit_audio is off
    assert not [f for f in out if f["kind"] == "audio"]


def test_slice_emit_audio_produces_slice_frames(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_click_train(), SR)
    out = edit.slice_onsets(src, emit_audio=True)
    slices = [f for f in out if f["kind"] == "audio"]
    assert slices, "expected sliced audio frames"
    for i, f in enumerate(slices):
        assert f["role"].startswith("slice:")
        assert f.get("of") == src["id"]
        assert f.get("lineage") == [src["id"]]
        assert f.get("op") == "slice" and f.get("op_version") == edit.SLICE_OP_VERSION
        assert "start_sample" in f["params"] and "end_sample" in f["params"]


def test_slice_silence_yields_empty_markers(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(np.zeros((SR, 1), dtype=np.float32), SR)
    out = edit.slice_onsets(src, emit_audio=True)
    markers = [f for f in out if f["kind"] == "marker"]
    assert len(markers) == 1
    assert markers[0]["data"] == []
    assert not [f for f in out if f["kind"] == "audio"]  # nothing to slice


# --- select (stream filter, smplstream.select) -----------------------------------------


def test_select_last_wins():
    from smplstream import select as S

    frames = [
        {"id": "a", "kind": "audio", "role": "stem:drums"},
        {"id": "b", "kind": "audio", "role": "stem:drums"},
        {"id": "c", "kind": "audio", "role": "stem:bass"},
    ]
    got = S.select(frames, role="stem:drums", mode="last")
    assert [f["id"] for f in got] == ["b"]  # most-recent match


def test_select_all_returns_every_match():
    from smplstream import select as S

    frames = [
        {"id": "a", "kind": "audio", "role": "stem:drums"},
        {"id": "b", "kind": "audio", "role": "stem:drums"},
    ]
    got = S.select(frames, role="stem:drums", mode="all")
    assert [f["id"] for f in got] == ["a", "b"]


def test_select_strict_errors_on_multiple():
    from smplstream import select as S
    from smplstream.errors import ResolutionError

    frames = [
        {"id": "a", "kind": "audio", "role": "stem:drums"},
        {"id": "b", "kind": "audio", "role": "stem:drums"},
    ]
    with pytest.raises(ResolutionError):
        S.select(frames, role="stem:drums", mode="strict")


def test_select_by_kind():
    from smplstream import select as S

    frames = [
        {"id": "a", "kind": "audio", "role": "source"},
        {"id": "b", "kind": "feature", "role": "summary"},
        {"id": "c", "kind": "feature", "role": "qc"},
    ]
    got = S.select(frames, kind="feature", mode="all")
    assert [f["id"] for f in got] == ["b", "c"]
