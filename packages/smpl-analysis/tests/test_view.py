"""Tests for smpl_analysis.view — consolidated multimodal report (ticket vault-31r9).

Verifies the report frame's kind/role/op_version/lineage, that the markdown tables
every feature key with a unit, counts + previews markers, lists images with their
resolved CAS path, surfaces errors prominently, and round-trips oversized reports
to the CAS.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from smplstream import cas, error_frame, frames as F

from smpl_analysis import view as V


def _tone(sr: int = 22050, dur: float = 0.5, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture()
def cas_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    return tmp_path


@pytest.fixture()
def audio_frame(tmp_path, cas_env):
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), _tone(), 22050, subtype="FLOAT")
    h = cas.put_audio_file(str(wav))
    meta = cas.read_meta(h) or {}
    return F.audio_frame(
        h, sr=meta.get("sr", 22050), ch=meta.get("ch", 1), dur=meta.get("dur", 0.5),
        role="source", op="read", op_version="read@1",
    )


@pytest.fixture()
def image_frame(audio_frame, cas_env):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 256  # a stub PNG blob (bytes are CAS-opaque here)
    h = cas.put_blob(png, "image/png")
    return F.image_frame(
        h, role="spectrogram:mel", of=audio_frame["id"], op="spectrogram",
        op_version="spectrogram@1",
    )


def _report_frame(frames):
    out = V.view_frames(frames)
    assert len(out) == 1
    return out[0]


# --- build_report (pure, no CAS for the non-image paths) ---

def test_report_lists_feature_keys_with_units():
    feat = F.feature_frame(
        {
            "loudness.integrated_lufs": -14.2,
            "lowlevel.spectral_flatness_db": {"mean": -22.5, "stdev": 3.1},
            "timbre.brightness": 61,
        },
        role="loudness", of="blake3:" + "a" * 64, op="loudness", op_version="loudness@1",
    )
    md = V.build_report([feat])
    # every key appears, with its unit derived from the namespaced/suffixed spelling
    assert "`loudness.integrated_lufs`" in md
    assert "LUFS" in md
    assert "`lowlevel.spectral_flatness_db`" in md
    assert "dB" in md
    # the {mean, stdev} statistic convention renders compactly
    assert "(±3.1)" in md or "(±3.1000)".rstrip("0") in md
    assert "`timbre.brightness`" in md and "0–100" in md


def test_report_lists_markers_count_and_first_times():
    pts = [{"t": round(i * 0.25, 6), "sample": i * 5512, "label": "onset"} for i in range(8)]
    mk = F.marker_frame(pts, role="onset", of="blake3:" + "b" * 64, op="onsets")
    md = V.build_report([mk])
    assert "## Markers" in md
    assert "8 point(s)" in md
    # first few times spelled out, the tail elided
    assert "0s" in md and "0.25s" in md
    assert "+3" in md  # 8 points, preview 5 -> "+3" elided


def test_report_surfaces_errors_prominently():
    err = error_frame("op_failed", "librosa blew up", of="blake3:" + "c" * 64, op="cat")
    feat = F.feature_frame({"loudness.integrated_lufs": -10.0}, role="loudness", op="loudness")
    md = V.build_report([err, feat])
    # errors section comes before features
    assert md.index("Errors") < md.index("Features")
    assert "op_failed" in md
    assert "librosa blew up" in md


def test_report_lists_images_with_resolved_cas_path(audio_frame, image_frame):
    md = V.build_report([audio_frame, image_frame])
    assert "## Images" in md
    assert "spectrogram:mel" in md
    # the resolved CAS path for the PNG must appear so an LLM can open it
    path = str(cas.get_path(image_frame["hash"]))
    assert path in md


def test_report_handles_unresolvable_image_gracefully(cas_env):
    # an image frame whose blob isn't in the CAS must degrade, not crash
    ghost = F.image_frame("blake3:" + "f" * 64, role="spectrogram:mel", op="spectrogram")
    md = V.build_report([ghost])
    assert "## Images" in md
    assert "unresolved" in md


def test_empty_stream_is_valid():
    md = V.build_report([])
    assert "# smpl analysis report" in md
    assert "empty stream" in md
    out = V.view_frames([])
    assert len(out) == 1
    assert out[0]["kind"] == "text" and out[0]["role"] == "report"


# --- view_frames (the derived frame contract) ---

def test_view_frame_contract(audio_frame, image_frame):
    feat = F.feature_frame({"loudness.integrated_lufs": -14.0}, role="loudness",
                           of=audio_frame["id"], op="loudness", op_version="loudness@1")
    frames = [audio_frame, image_frame, feat]
    rf = _report_frame(frames)

    assert rf["kind"] == "text"
    assert rf["role"] == "report"
    assert rf["op"] == "view"
    assert rf["op_version"] == "view@1"
    # report anchored to the audio frame, lineage closes over every input id
    assert rf["of"] == audio_frame["id"]
    assert set(rf["lineage"]) == {f["id"] for f in frames}
    # inline (small) report
    assert isinstance(rf["data"], str)
    assert rf["data"].startswith("# smpl analysis report")
    # structurally valid per the protocol
    assert F.validate_frame(rf) == []


def test_oversized_report_moves_to_cas(cas_env):
    # Many feature frames -> a report well over 64 KiB -> MUST move to CAS.
    big = []
    for i in range(4000):
        big.append(
            F.feature_frame(
                {f"lowlevel.spectral_flatness_db": {"mean": -20.0 - i * 0.001, "stdev": 1.234}},
                role="loudness", of="blake3:" + "d" * 64, op="spectral", op_version="spectral@1",
            )
        )
    rf = _report_frame(big)
    assert rf["kind"] == "text" and rf["role"] == "report"
    # oversized -> hash, not inline data
    assert "hash" in rf and "data" not in rf
    assert rf["media"] == "text/plain"
    assert F.validate_frame(rf) == []
    # the stored blob is the full markdown report
    stored = cas.get_path(rf["hash"]).read_text()
    assert stored.startswith("# smpl analysis report")
    assert "## Features" in stored
