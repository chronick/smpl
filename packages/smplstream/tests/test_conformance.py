"""Conformance suite: the three invariants + the golden-hash corpus.

The golden corpus pins canonical-PCM stability: deterministic signals → fixed expected
hashes. If a libsndfile/numpy/blake3 change ever shifts a decode at the bit level, these
fail loudly (the whole point — a passthrough-only test would miss it).
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from smplstream import conformance, error_frame, frames as F, hashing


# ---- golden-hash corpus -------------------------------------------------------------
# Generated once from the reference decoder (libsndfile via soundfile) and frozen.

def _signal(kind: str, sr: int, n: int) -> np.ndarray:
    t = np.arange(n) / sr
    if kind == "silence":
        return np.zeros(n, dtype=np.float32)
    if kind == "dc":
        return np.full(n, 0.25, dtype=np.float32)
    if kind == "sine":
        return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    if kind == "ramp":
        return np.linspace(-1.0, 1.0, n, dtype=np.float32)
    raise ValueError(kind)


GOLDEN = {
    # (kind, sr, n, subtype): expected blake3 audio hash. Frozen from the reference decoder
    # (libsndfile via soundfile). Regenerate with `python -m smplstream._gen_golden`.
    ("silence", 8000, 800, "FLOAT"): "blake3:85157125b381a2aa60b4db1a5f4336f6d722e4df25dc64f9028e81ff93a033ea",
    ("dc", 8000, 800, "FLOAT"): "blake3:e4b450d61d3282fe2cff6918fee8fe12cc447ff02ab2fd93eb3607edad510a21",
    ("sine", 8000, 800, "FLOAT"): "blake3:2f5c977cafeed3521597ce198e6d5c5a592c0cee32f9e60d7197723c777fe39d",
    ("ramp", 8000, 800, "FLOAT"): "blake3:47a9498d3c25b344e4d8dcce9cc922accac94087e6913a7cf1529763b5af54aa",
}


def _wav(kind, sr, n, subtype):
    buf = io.BytesIO()
    sf.write(buf, _signal(kind, sr, n), sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def test_golden_hashes_match_frozen_corpus():
    """Each golden signal MUST hash to its frozen constant — catches a silent decoder drift
    (libsndfile/numpy/blake3) that a passthrough-only test would miss."""
    for (kind, sr, n, subtype), expected in GOLDEN.items():
        wav = _wav(kind, sr, n, subtype)
        got = hashing.audio_hash_bytes(wav)
        assert got == expected, f"golden drift for {kind}: {got} != {expected}"
        # Both hashing routes (file/bytes and raw-PCM) must agree.
        pcm, srr, ch = hashing.decode_canonical_bytes(wav)
        assert hashing.audio_hash_from_pcm(pcm, srr, ch) == expected


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


def test_clean_stream_passes_all():
    a = F.audio_frame("blake3:" + "a" * 64, sr=48000, ch=1, dur=1.0, role="source")
    b = F.text_frame("caption text", role="caption", of=a["id"])
    out = [a, b]
    assert conformance.check_all([a], out) == []
