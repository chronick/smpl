"""Tests for smpl_analysis.loudness (ticket vault-3vau).

Verifies the three registered feature keys, units/op_version contract, true-peak-over
markers (with sample: indices), and that the integrated-LUFS measurement tracks gain.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from smpl_analysis import loudness as L


def _tone(sr=48000, freq=1000.0, dur=4.0, amp=0.5, ch=1):
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    sig = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if ch == 1:
        return sig[:, None], sr
    return np.column_stack([sig] * ch), sr


def test_analyze_array_keys_and_types():
    data, sr = _tone()
    res = L.analyze_array(data, sr)
    assert set(res) >= {"integrated_lufs", "true_peak_dbtp", "max_short_term_lufs", "over_points", "sr"}
    assert res["sr"] == sr
    assert math.isfinite(res["integrated_lufs"])
    assert math.isfinite(res["true_peak_dbtp"])
    assert math.isfinite(res["max_short_term_lufs"])
    # A -6 dBFS sine has a true peak right around -6 dBTP (no inter-sample lift at 1 kHz).
    assert -7.5 < res["true_peak_dbtp"] < -4.5
    # Sustained tone: integrated and short-term peak should be close.
    assert abs(res["integrated_lufs"] - res["max_short_term_lufs"]) < 1.5


def test_integrated_lufs_tracks_gain():
    loud, sr = _tone(amp=0.5)
    quiet, _ = _tone(amp=0.05)
    rl = L.analyze_array(loud, sr)["integrated_lufs"]
    rq = L.analyze_array(quiet, sr)["integrated_lufs"]
    # 20*log10(0.5/0.05) = 20 dB quieter, ~20 LU lower.
    assert rl - rq == pytest.approx(20.0, abs=1.0)


def test_silence_is_none_in_frame():
    data = np.zeros((48000 * 4, 1), dtype=np.float32)
    sr = 48000
    af = {"id": "blake3:" + "0" * 64, "hash": "x"}
    # call the array path directly (no CAS) to exercise the silence -> None mapping
    res = L.analyze_array(data, sr)
    assert not math.isfinite(res["true_peak_dbtp"])  # -inf
    # _db_round maps -inf -> None
    assert L._db_round(res["true_peak_dbtp"]) is None
    assert L._db_round(res["integrated_lufs"]) is None


def test_loudness_frames_via_cas(tmp_path, monkeypatch):
    import soundfile as sf
    from smplstream import cas, frames as F

    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))

    # Build a tone with a deliberate near-full-scale transient to trigger a true-peak over.
    data, sr = _tone(amp=0.5)
    data = data.copy()
    data[1000, 0] = 0.999  # a single hot sample near 0 dBFS -> over the -1 dBTP ceiling
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), data, sr, subtype="FLOAT")

    h = cas.put_audio_file(str(wav))
    meta = cas.read_meta(h) or {}
    audio = F.audio_frame(h, sr=meta.get("sr", sr), ch=meta.get("ch", 1),
                          dur=meta.get("dur", 0.0), role="source", op="read")

    out = L.loudness_frames(audio, emit_markers=True, over_ceiling_dbtp=-1.0)

    feats = [f for f in out if f["kind"] == "feature"]
    markers = [f for f in out if f["kind"] == "marker"]
    assert len(feats) == 1
    feat = feats[0]

    # Exact registered keys, units, op_version, lineage.
    assert set(feat["data"]) == {
        "loudness.integrated_lufs",
        "loudness.true_peak_dbtp",
        "loudness.max_short_term_lufs",
    }
    assert feat["op"] == "loudness"
    assert feat["op_version"] == "loudness@1"
    assert feat["of"] == audio["id"]
    assert feat["role"] == "loudness"

    # The hot sample should push true-peak over the ceiling and produce a marker.
    assert feat["data"]["loudness.true_peak_dbtp"] > -1.0
    assert len(markers) == 1
    mk = markers[0]
    assert mk["role"] == "true-peak-over"
    assert mk["of"] == audio["id"]
    assert mk["data"], "expected at least one over point"
    pt = mk["data"][0]
    assert "sample" in pt and isinstance(pt["sample"], int)
    assert "t" in pt


def test_no_markers_flag():
    data, sr = _tone(amp=0.5)
    res = L.analyze_array(data, sr, over_ceiling_dbtp=-1.0)
    # A clean -6 dBFS tone has no overs at the -1 dBTP ceiling.
    assert res["over_points"] == []
