"""Tests for smpl_analysis.spectrogram — PNG render + image-frame emission."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from smplstream import cas, frames as F

from smpl_analysis import spectrogram as spec


def _tone(sr: int = 22050, dur: float = 0.5, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture()
def cas_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    return tmp_path


@pytest.fixture()
def audio_frame(tmp_path, cas_dir):
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), _tone(), 22050, subtype="FLOAT")
    h = cas.put_audio_file(str(wav))
    meta = cas.read_meta(h) or {}
    return F.audio_frame(
        h, sr=meta.get("sr", 22050), ch=meta.get("ch", 1), dur=meta.get("dur", 0.5),
        role="source", op="read", op_version="read@1",
    )


def _is_png(b: bytes) -> bool:
    return b[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("kind", ["mel", "cqt", "hpss", "waveform"])
def test_render_array_emits_png(kind):
    y = _tone()
    png = spec.render_array(y, 22050, kind)
    assert _is_png(png)
    assert len(png) > 1000  # a real plot, not an empty canvas


def test_render_array_rejects_unknown_kind():
    with pytest.raises(ValueError):
        spec.render_array(_tone(), 22050, "bogus")


def test_default_kind_is_mel(audio_frame):
    out = spec.render_audio_frame(audio_frame)
    assert len(out) == 1
    f = out[0]
    assert f["kind"] == "image"
    assert f["role"] == "spectrogram:mel"
    assert f["media"] == "image/png"
    assert f["of"] == audio_frame["id"]
    assert f["op"] == "spectrogram"
    assert f["op_version"] == "spectrogram@1"
    assert f["params"]["kind"] == "mel"
    # the referenced PNG actually landed in the CAS
    assert _is_png(cas.get_path(f["hash"]).read_bytes())


def test_all_kinds_roles_and_lineage(audio_frame):
    out = spec.render_audio_frame(audio_frame, kinds=["mel", "cqt", "hpss", "waveform"])
    roles = [f["role"] for f in out]
    assert roles == ["spectrogram:mel", "spectrogram:cqt", "spectrogram:hpss", "waveform"]
    for f in out:
        assert f["kind"] == "image"
        assert f["of"] == audio_frame["id"]
        assert f["op_version"] == "spectrogram@1"
        assert _is_png(cas.get_path(f["hash"]).read_bytes())


def test_image_frames_are_structurally_valid(audio_frame):
    out = spec.render_audio_frame(audio_frame, kinds=["mel", "waveform"])
    for f in out:
        assert F.validate_frame(f) == []
