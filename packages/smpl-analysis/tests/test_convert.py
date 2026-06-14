"""Tests for `smpl_analysis.convert` — ffmpeg format/sr/bit-depth conversion as an explicit op.

The conversion shells out to ffmpeg; tests are skipped if ffmpeg is not on PATH. They run
against an isolated CAS (SMPL_CAS_DIR -> tmp) so they never touch the real store.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import soundfile as sf

ffmpeg_missing = shutil.which("ffmpeg") is None
pytestmark = pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg not on PATH")


def _seed_audio_frame(cas_dir, *, sr=48000, dur=0.25, ch=1):
    """Write a short tone, CAS it, return (input audio frame dict, its hash)."""
    import os

    os.environ["SMPL_CAS_DIR"] = str(cas_dir)
    from smplstream import cas, frames as F

    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    if ch == 2:
        tone = np.stack([tone, tone], axis=1)
    wav = cas_dir / "tone.wav"
    sf.write(str(wav), tone, sr, subtype="FLOAT")
    h = cas.put_audio_file(wav)
    meta = cas.read_meta(h) or {}
    frame = F.audio_frame(h, sr=meta.get("sr", sr), ch=meta.get("ch", ch),
                          dur=meta.get("dur", dur), role="source", op="read", fmt=meta.get("fmt"))
    return frame, h


def test_convert_resamples_to_new_frame_and_hash(tmp_path):
    cas_dir = tmp_path / "cas"
    cas_dir.mkdir()
    src_frame, src_hash = _seed_audio_frame(cas_dir, sr=48000)

    from smpl_analysis import convert

    out = convert.convert_audio_frame(src_frame, sr=44100, fmt="wav")
    assert len(out) == 1
    f = out[0]
    assert f["kind"] == "audio"
    # Explicit op, NOT a silent alias: new frame carries a DIFFERENT hash from the source.
    assert f["hash"] != src_hash
    assert f["meta"]["sr"] == 44100
    assert f["op"] == "convert"
    assert f["op_version"] == "convert@1"
    assert f["role"] == "converted"
    # Lineage points back at the input id; of is set too.
    assert f["lineage"] == [src_frame["id"]]
    assert f["of"] == src_frame["id"]


def test_params_capture_target_and_env_fingerprint(tmp_path):
    cas_dir = tmp_path / "cas"
    cas_dir.mkdir()
    src_frame, _ = _seed_audio_frame(cas_dir, sr=48000)

    from smpl_analysis import convert

    out = convert.convert_audio_frame(src_frame, sr=44100, bits=24, fmt="flac")
    f = out[0]
    p = f["params"]
    assert p["format"] == "flac"
    assert p["sr"] == 44100
    assert p["bits"] == 24
    assert p["input_hash"] == src_frame["hash"]
    # env_fingerprint binds the ffmpeg version into the memo basis (spec → Memoization).
    assert isinstance(p["env_fingerprint"], str) and len(p["env_fingerprint"]) == 64
    assert f["media"] == "audio/flac"


def test_same_request_is_deterministic_same_hash(tmp_path):
    cas_dir = tmp_path / "cas"
    cas_dir.mkdir()
    src_frame, _ = _seed_audio_frame(cas_dir, sr=48000)

    from smpl_analysis import convert

    a = convert.convert_audio_frame(src_frame, sr=44100, fmt="wav")[0]
    b = convert.convert_audio_frame(src_frame, sr=44100, fmt="wav")[0]
    # Same input + same target -> same canonical-PCM hash -> same content id (dedup).
    assert a["hash"] == b["hash"]
    assert a["id"] == b["id"]


def test_unsupported_format_emits_error_frame(tmp_path):
    cas_dir = tmp_path / "cas"
    cas_dir.mkdir()
    src_frame, _ = _seed_audio_frame(cas_dir, sr=48000)

    from smpl_analysis import convert

    out = convert.convert_audio_frame(src_frame, fmt="ogg")
    assert len(out) == 1
    f = out[0]
    assert f["kind"] == "error"
    assert f["data"]["code"] == "unsupported"
    assert f["data"]["of"] == src_frame["id"]


def test_unsupported_bit_depth_emits_error_frame(tmp_path):
    cas_dir = tmp_path / "cas"
    cas_dir.mkdir()
    src_frame, _ = _seed_audio_frame(cas_dir, sr=48000)

    from smpl_analysis import convert

    out = convert.convert_audio_frame(src_frame, bits=12)
    assert out[0]["kind"] == "error"
    assert out[0]["data"]["code"] == "unsupported"


def test_missing_hash_input_errors(tmp_path):
    cas_dir = tmp_path / "cas"
    cas_dir.mkdir()
    import os

    os.environ["SMPL_CAS_DIR"] = str(cas_dir)
    from smpl_analysis import convert

    out = convert.convert_audio_frame({"id": "blake3:" + "0" * 64, "kind": "audio", "meta": {}})
    assert out[0]["kind"] == "error"
    assert out[0]["data"]["code"] == "unsupported"
