"""Tests for smpl_analysis.duo — the first 2-audio-input op family (ticket vault-nkvw).

Covers the ``vocode`` channel vocoder: modulator-envelope tracking, carrier-timbre transfer,
determinism, sample-rate mismatch, and the 2-input lineage convention (lineage carries BOTH
the carrier and modulator ids).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import duo


SR = 44100


# ---------------------------------------------------------------------------
# CAS isolation — a fresh SMPL_CAS_DIR per test (cas_dir() reads the env each call).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _cas_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))


def _put_wav(samples, sr, role="source"):
    from smplstream import cas, frames as F

    samples = np.asarray(samples, dtype="float32")
    if samples.ndim == 1:
        samples = samples[:, None]
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


def _load_mono(frame):
    from smplstream import cas

    data, sr = sf.read(str(cas.get_path(frame["hash"])), dtype="float64", always_2d=True)
    return data.mean(axis=1), sr


def _harmonic_carrier(f0=60.0, n_harm=40, dur=1.5, sr=SR):
    """A harmonic-rich tone at ``f0`` with 1/k-weighted harmonics (fundamental dominates)."""
    t = np.arange(int(dur * sr)) / sr
    sig = np.zeros_like(t)
    for k in range(1, n_harm + 1):
        f = f0 * k
        if f >= 0.45 * sr:
            break
        sig += (1.0 / k) * np.sin(2 * np.pi * f * t)
    sig /= np.max(np.abs(sig))
    return (0.8 * sig).astype("float32")


def _noise_bursts(dur=1.5, sr=SR, n_bursts=3, seed=1234):
    """``n_bursts`` white-noise bursts separated by silence — spanning the full ``dur``."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    sig = np.zeros(n, dtype="float64")
    seg = n // (2 * n_bursts)  # alternating burst / gap of equal length
    burst_ranges = []
    for b in range(n_bursts):
        start = (2 * b) * seg
        end = start + seg
        sig[start:end] = rng.standard_normal(end - start)
        burst_ranges.append((start, end))
    sig *= 0.5
    return sig.astype("float32"), burst_ranges, seg


def _short_time_rms(x, win):
    """Non-overlapping short-time RMS frames."""
    n = (len(x) // win) * win
    frames = x[:n].reshape(-1, win)
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-20)


# ---------------------------------------------------------------------------
# 1. Envelope tracking: output loudness follows the modulator; gaps go quiet.
# ---------------------------------------------------------------------------
def test_envelope_tracking_follows_modulator():
    carrier = _put_wav(_harmonic_carrier(), SR, role="carrier")
    mod_sig, burst_ranges, seg = _noise_bursts()
    modulator = _put_wav(mod_sig, SR, role="mod")

    wet = duo.apply_vocode(carrier, modulator)
    out, sr = _load_mono(wet)

    # Short-time RMS of output should correlate with the modulator's short-time RMS.
    win = 2048
    out_rms = _short_time_rms(out, win)
    mod_rms = _short_time_rms(mod_sig.astype("float64"), win)
    m = min(len(out_rms), len(mod_rms))
    r = np.corrcoef(out_rms[:m], mod_rms[:m])[0, 1]
    assert r > 0.7, f"output RMS envelope correlation with modulator too low: r={r:.3f}"

    # Gap energy should be a small fraction of burst energy.
    burst_e = 0.0
    for (s, e) in burst_ranges:
        burst_e += float(np.sum(out[s:e] ** 2))
    gap_e = 0.0
    for b in range(len(burst_ranges) - 1):
        gs = burst_ranges[b][1]
        ge = burst_ranges[b + 1][0]
        gap_e += float(np.sum(out[gs:ge] ** 2))
    assert gap_e < 0.10 * burst_e, f"gap energy {gap_e:.4g} not < 10% of burst energy {burst_e:.4g}"


# ---------------------------------------------------------------------------
# 2. Timbre from the carrier: output spectrum concentrated at carrier harmonics (low),
#    not at the broadband modulator noise.
# ---------------------------------------------------------------------------
def test_timbre_comes_from_carrier():
    carrier = _put_wav(_harmonic_carrier(), SR, role="carrier")
    # Sustained (full-length) red noise → roughly constant, low-tilted envelope (the WRUM case:
    # a deep-voiced modulator), so the output spectrum is dominated by the carrier's low harmonic
    # line structure rather than by modulator dynamics.
    rng = np.random.default_rng(7)
    red = np.cumsum(rng.standard_normal(int(1.5 * SR)))
    red -= red.mean()
    red = (0.5 * red / np.max(np.abs(red))).astype("float32")
    modulator = _put_wav(red, SR, role="mod")

    # ess_mix=0 isolates the vocoder BODY: the ess path deliberately injects fixed HF energy
    # (consonant intelligibility), which would otherwise swamp the carrier's low harmonics here.
    wet = duo.apply_vocode(carrier, modulator, ess_mix=0.0)
    out, sr = _load_mono(wet)

    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    freqs = np.fft.rfftfreq(len(out), 1.0 / sr)
    peak_freq = freqs[int(np.argmax(spec))]
    # Dominant peak is low AND sits on a carrier harmonic (multiple of 60 Hz) — the line
    # structure is the carrier's timbre, not the continuous-spectrum modulator noise.
    assert peak_freq < 200.0, f"dominant spectral peak at {peak_freq:.1f} Hz, expected < 200 Hz"
    assert abs(peak_freq % 60.0) < 6.0 or abs((peak_freq % 60.0) - 60.0) < 6.0, (
        f"dominant peak {peak_freq:.1f} Hz is not on a carrier harmonic (multiple of 60)")


# ---------------------------------------------------------------------------
# 3. Determinism: two runs are bit-identical.
# ---------------------------------------------------------------------------
def test_determinism():
    carrier = _put_wav(_harmonic_carrier(dur=1.0), SR, role="carrier")
    mod_sig, _, _ = _noise_bursts(dur=1.0)
    modulator = _put_wav(mod_sig, SR, role="mod")

    a = duo.apply_vocode(carrier, modulator)
    b = duo.apply_vocode(carrier, modulator)
    assert a["hash"] == b["hash"]
    assert a["id"] == b["id"]
    oa, _ = _load_mono(a)
    ob, _ = _load_mono(b)
    assert np.array_equal(oa, ob)


# ---------------------------------------------------------------------------
# 4. Sample-rate mismatch: a 22050 Hz modulator is resampled to the carrier's sr.
# ---------------------------------------------------------------------------
def test_sr_mismatch_modulator_resampled():
    carrier = _put_wav(_harmonic_carrier(dur=1.0), SR, role="carrier")
    mod_sig, _, _ = _noise_bursts(dur=1.0, sr=22050)
    modulator = _put_wav(mod_sig, 22050, role="mod")

    wet = duo.apply_vocode(carrier, modulator)
    assert wet["meta"]["sr"] == SR
    out, sr = _load_mono(wet)
    assert sr == SR
    assert np.max(np.abs(out)) > 0.0
    assert wet["params"]["sr_hz"] == SR


# ---------------------------------------------------------------------------
# 5. Lineage: the wet frame's lineage carries BOTH input ids (2-input convention).
# ---------------------------------------------------------------------------
def test_lineage_carries_both_inputs():
    carrier = _put_wav(_harmonic_carrier(dur=0.5), SR, role="carrier")
    mod_sig, _, _ = _noise_bursts(dur=0.5)
    modulator = _put_wav(mod_sig, SR, role="mod")

    wet = duo.apply_vocode(carrier, modulator)
    assert wet["kind"] == "audio"
    assert wet["role"] == "carrier.wet"
    assert wet.get("of") == carrier["id"]
    assert wet.get("lineage") == [carrier["id"], modulator["id"]]
    assert wet.get("op") == "vocode"
    assert wet.get("op_version") == duo.VOCODE_OP_VERSION
    assert carrier["id"] in wet["lineage"] and modulator["id"] in wet["lineage"]
    # vocode@2 contract: BODY normalized to 0.9, then the ess add may raise the final
    # peak up to the 0.98 safety cap (intelligibility fix — vowels no longer crushed).
    out, _ = _load_mono(wet)
    assert 0.85 < np.max(np.abs(out)) <= 0.98 + 1e-4
