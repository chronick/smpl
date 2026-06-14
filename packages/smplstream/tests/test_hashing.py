"""Canonical-PCM hashing invariants (spec → *Canonical PCM*, NORMATIVE)."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from smplstream import hashing


def _wav_bytes(samples, sr, subtype="FLOAT"):
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def _tone(freq=440.0, sr=48000, dur=0.25, ch=1):
    t = np.arange(int(sr * dur)) / sr
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if ch == 1:
        return mono
    return np.stack([mono] * ch, axis=1)


def test_hash_is_deterministic():
    wav = _wav_bytes(_tone(), 48000)
    assert hashing.audio_hash_bytes(wav) == hashing.audio_hash_bytes(wav)


def test_gain_changes_pcm_so_changes_hash():
    # "Canonical" means format-canonical, NOT level-normalized: gain MUST change the hash.
    a = _tone()
    b = (a * 0.5).astype(np.float32)
    assert hashing.audio_hash_bytes(_wav_bytes(a, 48000)) != hashing.audio_hash_bytes(_wav_bytes(b, 48000))


def test_sample_rate_binds_into_key():
    a = _tone(sr=44100)
    # Same sample VALUES, different declared rate → different hash (format identity binds).
    h1 = hashing.audio_hash_from_pcm(hashing.canonical_pcm_bytes(a), 44100, 1)
    h2 = hashing.audio_hash_from_pcm(hashing.canonical_pcm_bytes(a), 48000, 1)
    assert h1 != h2


def test_channels_bind_into_key():
    a = _tone(ch=1)
    pcm = hashing.canonical_pcm_bytes(a)
    assert hashing.audio_hash_from_pcm(pcm, 48000, 1) != hashing.audio_hash_from_pcm(pcm, 48000, 2)


def test_float_roundtrip_preserves_hash_across_containers():
    # FLOAT WAV and FLOAT AIFF of identical samples decode to identical PCM → identical hash.
    samples = _tone()
    wav = _wav_bytes(samples, 48000, "FLOAT")
    buf = io.BytesIO()
    sf.write(buf, samples, 48000, format="AIFF", subtype="FLOAT")
    aiff = buf.getvalue()
    assert hashing.audio_hash_bytes(wav) == hashing.audio_hash_bytes(aiff)


def test_int16_and_float_differ():
    # int16 quantizes; float doesn't → different canonical PCM → different hash (correct).
    samples = _tone()
    assert hashing.audio_hash_bytes(_wav_bytes(samples, 48000, "FLOAT")) != hashing.audio_hash_bytes(
        _wav_bytes(samples, 48000, "PCM_16")
    )


def test_hash_format():
    h = hashing.audio_hash_bytes(_wav_bytes(_tone(), 48000))
    assert h.startswith("blake3:") and len(h.split(":")[1]) == 64


def test_canonical_bytes_are_little_endian_float32():
    samples = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    raw = hashing.canonical_pcm_bytes(samples)
    back = np.frombuffer(raw, dtype="<f4")
    assert np.allclose(back, samples)


def test_pcm_length_must_match_channels():
    # 5 bytes can't form whole stereo float32 frames (multiple of 8) → reject.
    with pytest.raises(ValueError):
        hashing.audio_hash_from_pcm(b"\x00" * 5, 48000, 2)


def test_rejects_3d_array():
    with pytest.raises(ValueError):
        hashing.canonical_pcm_bytes(np.zeros((4, 2, 3), dtype=np.float32))  # via _channels guard? no
    with pytest.raises(ValueError):
        hashing._channels(np.zeros((4, 2, 3), dtype=np.float32))
