"""Frame construction, id assignment, and structural validation."""

from __future__ import annotations

from smplstream import frames as F
from smplstream import error_frame, validate_frame
from smplstream.ids import content_id, mint_id


def test_audio_frame_valid():
    fr = F.audio_frame("blake3:" + "a" * 64, sr=48000, ch=2, dur=1.0, role="source")
    assert validate_frame(fr) == []
    assert fr["v"] == 1
    assert fr["id"].startswith("blake3:")


def test_content_id_is_stable_and_dedups():
    f1 = F.audio_frame("blake3:" + "a" * 64, sr=48000, ch=2, dur=1.0, role="source")
    f2 = F.audio_frame("blake3:" + "a" * 64, sr=48000, ch=2, dur=1.0, role="source")
    assert f1["id"] == f2["id"]  # same defining fields → same id → dedup


def test_no_sequential_ids():
    # Ids must be content/random tokens, never f1/f2 counters.
    fr = F.text_frame("hi", role="caption")
    assert not fr["id"].startswith("f")
    assert fr["id"].startswith("blake3:")


def test_hash_xor_data():
    bad = {"v": 1, "kind": "audio", "id": "x", "hash": "blake3:" + "a" * 64,
           "media": "audio/wav", "data": "oops"}
    assert any("mutually exclusive" in p for p in validate_frame(bad))


def test_media_required_with_hash():
    bad = {"v": 1, "kind": "audio", "id": "x", "hash": "blake3:" + "a" * 64}
    assert any("media" in p for p in validate_frame(bad))


def test_vector_dim_gt_64_must_be_cas():
    bad = {"v": 1, "kind": "vector", "id": "x", "data": [0.0] * 128, "meta": {"dim": 128}}
    assert any("CAS" in p for p in validate_frame(bad))


def test_error_frame_requires_valid_code():
    fr = error_frame("decode_failed", "boom", of="blake3:" + "a" * 64, op="read")
    assert validate_frame(fr) == []
    assert fr["data"]["code"] == "decode_failed"


def test_future_version_rejected():
    fr = {"v": 2, "kind": "audio", "id": "x", "hash": "blake3:" + "a" * 64, "media": "audio/wav"}
    assert any("exceeds supported" in p for p in validate_frame(fr))


def test_mint_id_preserves_existing():
    fr = {"v": 1, "kind": "text", "id": "blake3:" + "f" * 64, "data": "x"}
    assert mint_id(fr)["id"] == "blake3:" + "f" * 64  # passthrough keeps inbound id verbatim
