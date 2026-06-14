"""Canonical decoded-PCM hashing — the normative hash basis (spec → *Canonical PCM*).

The audio hash MUST be reproducible across decoders, library versions, and machines,
or the cache is silently non-portable. Audio is decoded to ONE canonical form with no
discretionary conversion before hashing:

  - IEEE float32, little-endian
  - native channel count, interleaved, file-declared order (NO down/upmix)
  - native sample rate (NO resampling)
  - NO amplitude normalization, NO dither, NO metadata

    audio_hash = blake3( canonical_pcm_bytes || u32le(sr) || u8(ch) || u8(format_tag) )

The reference decoder is libsndfile (via ``soundfile``). The golden-hash conformance
corpus is generated from it; a decoder that disagrees at the bit level is a conformance
bug, not a license to re-define the canonical form here.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
from blake3 import blake3

# Format tag for the canonical PCM encoding. Bumping the canonical format (should it
# ever change) changes this tag, which changes every audio hash — by design.
FORMAT_TAG_FLOAT32LE = 1

# Cap implied by the u8 channel field in the key. Real audio never approaches this.
_MAX_CHANNELS = 255


def _b3_hex(data: bytes) -> str:
    return blake3(data).hexdigest()


def canonical_pcm_bytes(samples: np.ndarray) -> bytes:
    """Layout a decoded sample array as canonical interleaved float32-LE bytes.

    Accepts mono ``(frames,)`` or multichannel ``(frames, channels)`` arrays (the
    ``soundfile`` native shape). C-contiguous ``(frames, channels)`` → interleaved.
    """
    arr = np.ascontiguousarray(samples)
    # Force IEEE float32 little-endian regardless of host byte order.
    le_f32 = arr.astype("<f4", copy=False)
    return le_f32.tobytes(order="C")


def _channels(samples: np.ndarray) -> int:
    return 1 if samples.ndim == 1 else int(samples.shape[1])


def audio_hash_from_pcm(pcm_bytes: bytes, sample_rate: int, channels: int) -> str:
    """Compute the canonical audio hash from already-canonical PCM bytes + format identity."""
    if not (1 <= channels <= _MAX_CHANNELS):
        raise ValueError(f"channel count {channels} out of range [1, {_MAX_CHANNELS}]")
    if not (0 < sample_rate < 2**32):
        raise ValueError(f"sample rate {sample_rate} out of range for u32")
    hasher = blake3()
    hasher.update(pcm_bytes)
    # Bind format identity so rate/channels/format can't alias to one key.
    hasher.update(struct.pack("<I", int(sample_rate)))
    hasher.update(struct.pack("<B", int(channels)))
    hasher.update(struct.pack("<B", FORMAT_TAG_FLOAT32LE))
    return "blake3:" + hasher.hexdigest()


def _decode_with_soundfile(target) -> tuple[np.ndarray, int]:
    import soundfile as sf

    # dtype=float32 → libsndfile's deterministic float conversion of the source PCM.
    # always_2d=False → native channel layout (mono stays 1-D, multichannel stays N-D).
    data, sr = sf.read(target, dtype="float32", always_2d=False)
    return data, int(sr)


def decode_canonical_file(path: str | Path) -> tuple[bytes, int, int]:
    """Decode an audio file to ``(canonical_pcm_bytes, sample_rate, channels)``."""
    data, sr = _decode_with_soundfile(str(path))
    return canonical_pcm_bytes(data), sr, _channels(data)


def decode_canonical_bytes(blob: bytes) -> tuple[bytes, int, int]:
    """Decode in-memory encoded audio (e.g. a WAV byte stream) to canonical PCM."""
    data, sr = _decode_with_soundfile(io.BytesIO(blob))
    return canonical_pcm_bytes(data), sr, _channels(data)


def audio_hash_file(path: str | Path) -> str:
    pcm, sr, ch = decode_canonical_file(path)
    return audio_hash_from_pcm(pcm, sr, ch)


def audio_hash_bytes(blob: bytes) -> str:
    pcm, sr, ch = decode_canonical_bytes(blob)
    return audio_hash_from_pcm(pcm, sr, ch)


def blob_hash(data: bytes) -> str:
    """Content hash for already-canonical blobs (PNG, .npy, MP4, JSON, text)."""
    return "blake3:" + _b3_hex(data)


def audio_duration_seconds(num_frames: int, sample_rate: int) -> float:
    return float(num_frames) / float(sample_rate) if sample_rate else 0.0


def probe_audio(path_or_bytes: str | Path | bytes | BinaryIO) -> dict:
    """Cheap metadata probe (sr, ch, frames, dur, fmt, subtype) without hashing.

    Used to populate ``meta`` and the CAS sidecar.
    """
    import soundfile as sf

    target = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else (
        str(path_or_bytes) if isinstance(path_or_bytes, (str, Path)) else path_or_bytes
    )
    info = sf.info(target)
    return {
        "sr": int(info.samplerate),
        "ch": int(info.channels),
        "frames": int(info.frames),
        "dur": audio_duration_seconds(info.frames, info.samplerate),
        "fmt": info.format,
        "subtype": info.subtype,
    }
