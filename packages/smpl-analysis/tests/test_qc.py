"""Tests for smpl_analysis.qc — deterministic QC top-6 (ticket vault-1e9a).

Each test synthesizes a signal with a KNOWN defect and asserts the corresponding QC measure
fires (and that clean signals don't). The end-to-end test exercises the frame-emitting path
through the CAS, asserting registry-correct keys and lineage.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import qc


SR = 44100


def _tone(freq=440.0, dur=1.0, amp=0.5, sr=SR, channels=1):
    t = np.arange(int(dur * sr)) / sr
    sig = amp * np.sin(2 * np.pi * freq * t)
    if channels == 1:
        return sig.reshape(-1, 1).astype(np.float32)
    return np.column_stack([sig] * channels).astype(np.float32)


# --- 1. clipping / true-peak ------------------------------------------------------------


def test_true_peak_clean_below_threshold():
    s = _tone(amp=0.5)
    tp = qc.true_peak_dbtp(s, SR)
    assert tp < qc._CLIP_DBTP_THRESHOLD  # −0.5 amp ≈ −6 dBTP, well clear


def test_true_peak_clipped_full_scale():
    s = _tone(amp=1.0)
    # hard-clip to full scale so inter-sample peaks exceed 0 dBTP
    s = np.clip(s * 4.0, -1.0, 1.0).astype(np.float32)
    tp = qc.true_peak_dbtp(s, SR)
    assert tp >= qc._CLIP_DBTP_THRESHOLD


# --- 2. phase correlation ---------------------------------------------------------------


def test_phase_correlation_mono_is_none():
    assert qc.phase_correlation(_tone(channels=1)) is None


def test_phase_correlation_in_phase_near_plus_one():
    s = _tone(channels=2)  # identical L/R
    corr = qc.phase_correlation(s)
    assert corr is not None and corr > 0.99


def test_phase_correlation_anti_phase_near_minus_one():
    s = _tone(channels=2)
    s[:, 1] *= -1.0  # invert right channel
    corr = qc.phase_correlation(s)
    assert corr is not None and corr < -0.99


# --- 3. DC offset -----------------------------------------------------------------------


def test_dc_offset_clean_is_very_low():
    s = _tone()
    assert qc.dc_offset_dbfs(s) < -60.0


def test_dc_offset_detects_bias():
    s = _tone(amp=0.3) + 0.1  # +0.1 DC bias (≈ −20 dBFS)
    s = s.astype(np.float32)
    dc = qc.dc_offset_dbfs(s)
    assert -25.0 < dc < -15.0


# --- 4. SNR -----------------------------------------------------------------------------


def _tone_with_silence(amp=0.5, tone_s=0.5, sil_s=0.3, sr=SR, noise=0.0):
    """A tone body bracketed by (optionally noisy) silence — the regime SNR is built for."""
    rng = np.random.default_rng(7)
    body = _tone(amp=amp, dur=tone_s, sr=sr)[:, 0]
    sil = np.zeros(int(sil_s * sr), dtype=np.float32)
    sig = np.concatenate([sil, body, sil]).astype(np.float64)
    if noise > 0:
        sig = sig + rng.normal(0, noise, sig.shape[0])
    return sig.reshape(-1, 1).astype(np.float32)


def test_snr_clean_signal_with_silence_is_high():
    # body well above a near-silent floor → wide dynamic range
    snr = qc.snr_db(_tone_with_silence(amp=0.5, noise=0.0), SR)
    assert snr is not None and snr > 30.0


def test_snr_noisy_floor_is_lower_than_clean():
    snr_clean = qc.snr_db(_tone_with_silence(amp=0.5, noise=1e-5), SR)
    snr_noisy = qc.snr_db(_tone_with_silence(amp=0.5, noise=0.02), SR)
    assert snr_noisy is not None and snr_clean is not None
    # raising the floor noise narrows the loud-to-quiet gap
    assert snr_noisy < snr_clean


# --- 5. clicks / gaps -------------------------------------------------------------------


def test_detect_clicks_finds_injected_spike():
    s = _tone(amp=0.3).copy()
    pos = SR // 2
    s[pos, 0] = 0.95  # a sharp discontinuity
    points = qc.detect_clicks(s, SR)
    assert points, "expected at least one click marker"
    # the click should land near the injected sample
    assert any(abs(p["sample"] - pos) <= 2 for p in points)
    assert all(p["label"] == "click" for p in points)


def test_detect_clicks_clean_tone_none():
    assert qc.detect_clicks(_tone(amp=0.3), SR) == []


def test_detect_gaps_finds_internal_silence():
    s = _tone(amp=0.4).copy()
    g0, g1 = SR // 3, SR // 3 + int(SR * 0.05)  # 50 ms internal gap
    s[g0:g1, 0] = 0.0
    points = qc.detect_gaps(s, SR)
    assert points, "expected an internal gap marker"
    assert any(abs(p["sample"] - g0) <= 2 for p in points)
    assert all(p["label"] == "gap" for p in points)


def test_detect_gaps_ignores_trailing_silence():
    s = _tone(amp=0.4, dur=0.5).copy()
    s = np.vstack([s, np.zeros((int(SR * 0.2), 1), dtype=np.float32)])  # trailing silence only
    assert qc.detect_gaps(s, SR) == []


# --- 6. lossy origin --------------------------------------------------------------------


def test_lossy_full_band_low_confidence():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 0.2, (SR, 1)).astype(np.float32)  # white noise → full-band
    res = qc.lossy_origin(s, SR)
    assert res["qc.lossy.expected_nyquist_hz"] == pytest.approx(SR / 2.0, abs=1.0)
    assert res["qc.lossy.confidence"] < 0.3


def test_lossy_brickwalled_flags_cutoff():
    # white noise low-passed hard at ~16 kHz, emulating a 128 kbps LAME brickwall
    rng = np.random.default_rng(2)
    from scipy.signal import butter, sosfiltfilt

    noise = rng.normal(0, 0.3, SR).astype(np.float64)
    sos = butter(12, 16000 / (SR / 2), btype="low", output="sos")
    band = sosfiltfilt(sos, noise).reshape(-1, 1).astype(np.float32)
    res = qc.lossy_origin(band, SR)
    assert res["qc.lossy.spectral_cutoff_hz"] < 18000.0
    assert res["qc.lossy.confidence"] > 0.3


# --- end-to-end: frame emission through the CAS -----------------------------------------


def _put_wav(samples, sr):
    """CAS a numpy buffer as a WAV and return its audio frame."""
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
        role="source",
    )


def test_qc_audio_frame_emits_feature_and_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    # stereo tone with an injected click so we also get a marker frame
    s = _tone(amp=0.4, channels=2).copy()
    s[SR // 2, 0] = 0.95
    af = _put_wav(s, SR)

    derived = qc.qc_audio_frame(af, want_markers=True)
    features = [f for f in derived if f["kind"] == "feature"]
    markers = [f for f in derived if f["kind"] == "marker"]

    assert len(features) == 1
    data = features[0]["data"]
    # registry keys present
    for key in (
        "qc.phase.correlation",
        "qc.dc_offset_dbfs",
        "qc.snr_db",
        "qc.lossy.spectral_cutoff_hz",
        "qc.lossy.expected_nyquist_hz",
        "qc.lossy.confidence",
    ):
        assert key in data, f"missing registry key {key}"
    # clipping flag + context true-peak
    assert "qc.clipping.detected" in data
    assert data["qc.clipping.detected"] is False  # −0.4 amp tone is clean
    # the loudness-owned KEY must NOT be emitted here (ownership boundary)
    assert "loudness.true_peak_dbtp" not in data

    # lineage / op metadata on every derived frame
    for f in derived:
        assert f.get("of") == af["id"]
        assert f.get("op") == "qc"
        assert f.get("op_version") == "qc@1"

    # the injected click yields a defect marker frame
    assert markers, "expected a marker frame for the injected click"
    assert any(p.get("label") == "click" for m in markers for p in m["data"])


def test_qc_audio_frame_no_markers(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(_tone(amp=0.4), SR)
    derived = qc.qc_audio_frame(af, want_markers=False)
    assert all(f["kind"] == "feature" for f in derived)
