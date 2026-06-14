"""Conformance suite: the four invariants + the golden-hash corpus.

The golden corpus pins canonical-PCM stability across the decode behaviors that actually
DRIFT (integer-subtype scaling, float64→float32 narrowing, multichannel interleave) — not
just near-passthrough FLOAT. If a libsndfile/numpy/blake3 change shifts a decode at the bit
level, these fail loudly (a passthrough-only test would miss it).
"""

from __future__ import annotations

import struct

import numpy as np

from smplstream import conformance, frames as F, hashing
from smplstream._gen_golden import wav_for_case

# Frozen from the reference decoder (libsndfile via soundfile).
# Regenerate with `python -m smplstream._gen_golden`.
GOLDEN = {
    ("silence", 8000, 800, "FLOAT"): "blake3:85157125b381a2aa60b4db1a5f4336f6d722e4df25dc64f9028e81ff93a033ea",
    ("dc", 8000, 800, "FLOAT"): "blake3:e4b450d61d3282fe2cff6918fee8fe12cc447ff02ab2fd93eb3607edad510a21",
    ("sine", 8000, 800, "FLOAT"): "blake3:2f5c977cafeed3521597ce198e6d5c5a592c0cee32f9e60d7197723c777fe39d",
    ("ramp", 8000, 800, "FLOAT"): "blake3:47a9498d3c25b344e4d8dcce9cc922accac94087e6913a7cf1529763b5af54aa",
    ("ramp", 8000, 800, "PCM_16"): "blake3:e6cf482438becee1a7de96750fab88cf09a5e7087bed3994c8f40f7f77f6ac95",
    ("ramp", 8000, 800, "PCM_24"): "blake3:a2e5fac038771448106c90db99da7d88cdf0d45c8ad5aaee286c08af2dad933b",
    ("third", 8000, 800, "DOUBLE"): "blake3:c11eb728fe5b32849563055863e8e2d544757c61a39252a577b8559abc8bfb16",
    ("stereo_lr", 8000, 800, "FLOAT"): "blake3:4e97447b519d2db88016d5408057bc344332c968c2183bd3cee703084d44a536",
}


def test_golden_hashes_match_frozen_corpus():
    """Each golden signal MUST hash to its frozen constant across int/double/stereo decodes."""
    for (kind, sr, n, subtype), expected in GOLDEN.items():
        wav = wav_for_case(kind, sr, n, subtype)
        got = hashing.audio_hash_bytes(wav)
        assert got == expected, f"golden drift for {kind}/{subtype}: {got} != {expected}"
        pcm, srr, ch = hashing.decode_canonical_bytes(wav)
        assert hashing.audio_hash_from_pcm(pcm, srr, ch) == expected


def test_golden_covers_drift_prone_decodes():
    """Guard against the corpus silently narrowing back to FLOAT-only mono."""
    subtypes = {s for (_, _, _, s) in GOLDEN}
    channels = {2 if k == "stereo_lr" else 1 for (k, _, _, _) in GOLDEN}
    assert {"PCM_16", "PCM_24", "DOUBLE", "FLOAT"} <= subtypes
    assert channels == {1, 2}  # at least one multichannel interleave case is pinned


def test_key_construction_order_is_pinned_independently():
    """Pin the exact concat order + format_tag by an INDEPENDENT construction — not just the
    opaque self-generated golden literals (which would move in lockstep with a layout bug)."""
    samples = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    pcm = hashing.canonical_pcm_bytes(samples)
    expected = (
        "blake3:"
        + __import__("blake3").blake3(
            pcm
            + struct.pack("<I", 8000)
            + struct.pack("<B", 1)
            + struct.pack("<B", hashing.FORMAT_TAG_FLOAT32LE)
        ).hexdigest()
    )
    assert hashing.audio_hash_from_pcm(pcm, 8000, 1) == expected


def test_passthrough_detects_dropped_frame():
    a = F.audio_frame("blake3:" + "a" * 64, sr=48000, ch=1, dur=1.0, role="source")
    b = F.text_frame("hi", role="caption", of=a["id"])
    problems = conformance.check_passthrough([a, b], [b])  # dropped `a`
    assert any("dropped" in p for p in problems)


def test_lineage_closure_detects_dangling():
    b = F.text_frame("hi", role="caption", of="blake3:" + "d" * 64)  # unresolvable `of`
    problems = conformance.check_lineage_closure([b])
    assert any("unresolvable" in p for p in problems)


def test_ordering_detects_forward_reference():
    a = F.audio_frame("blake3:" + "a" * 64, sr=48000, ch=1, dur=1.0, role="source")
    b = F.text_frame("hi", role="caption", of=a["id"])
    problems = conformance.check_ordering([b, a])  # b references a, but a is later
    assert any("appears later" in p for p in problems)


def test_id_collision_detected():
    a = F.audio_frame("blake3:" + "a" * 64, sr=48000, ch=1, dur=1.0, role="source")
    b = F.text_frame("different frame", role="caption")
    b["id"] = a["id"]  # force two DISTINCT frames to share an id
    problems = conformance.check_id_collisions([a, b])
    assert any("id_collision" in p for p in problems)


def test_id_dedup_is_not_a_collision():
    a = F.audio_frame("blake3:" + "a" * 64, sr=48000, ch=1, dur=1.0, role="source")
    a2 = F.audio_frame("blake3:" + "a" * 64, sr=48000, ch=1, dur=1.0, role="source")
    assert a["id"] == a2["id"]  # same frame, same content id
    assert conformance.check_id_collisions([a, a2]) == []  # dedup, not collision


def test_clean_stream_passes_all():
    a = F.audio_frame("blake3:" + "a" * 64, sr=48000, ch=1, dur=1.0, role="source")
    b = F.text_frame("caption text", role="caption", of=a["id"])
    assert conformance.check_all([a], [a, b]) == []
