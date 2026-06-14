"""`smpl convert` library — format / sample-rate / bit-depth conversion via ffmpeg.

`convert` is an **explicit op**, not a silent alias. It resolves the selected input audio
to a real file path, shells out to ffmpeg to re-encode at the requested container format,
sample rate, and bit depth, then `cas.put_audio_file`s the result under a *fresh* hash and
emits a new `audio` frame (`op: convert`, lineage → the input frame's id).

IMPORTANT — library/storage normalization is NOT the hash basis. The CAS audio hash is
always computed over canonical decoded PCM (native-rate / native-channel / float32, per
spec → *Canonical PCM*). Converting to a storage norm (e.g. 48 kHz / 24-bit for the music
library) therefore produces a *different* frame with a *different* hash; it does NOT change
how the source itself is hashed. Resampling and re-quantizing change the PCM, so the new
bytes correctly get a new content hash — that is the point of making `convert` an op.

The op records `env_fingerprint` (a hash of `ffmpeg -version`) inside `params` so the
memo key (spec → *Memoization*) tracks the ffmpeg version: a converter upgrade that alters
the resampler/dither output invalidates cached conversions instead of serving stale bytes.

Heavy / shell-out work stays inside the functions; nothing runs at import time.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

OP = "convert"
OP_VERSION = "convert@1"

# Container format -> (filename extension, frame `media` MIME). float32 default subtype keeps
# round-trips lossless when only the rate/container changes and no bit-depth was requested.
_FORMAT_EXT = {
    "wav": ("wav", "audio/wav"),
    "flac": ("flac", "audio/flac"),
    "aiff": ("aiff", "audio/aiff"),
    "aif": ("aiff", "audio/aiff"),
    "mp3": ("mp3", "audio/mpeg"),
}

# Requested PCM bit-depth -> ffmpeg sample_fmt / codec hints. ffmpeg picks the codec from the
# container, but we pin the sample format so "24-bit" actually lands as s32 (24 valid bits in
# WAV/FLAC) rather than silently staying float. None entries mean "let the container default".
_BITS_SAMPLE_FMT = {
    16: "s16",
    24: "s32",   # ffmpeg encodes 24-bit PCM via the 32-bit sample fmt (pcm_s24le for WAV)
    32: "s32",
}

# For WAV specifically, explicit pcm codecs give exact bit depth without ambiguity.
_WAV_BITS_CODEC = {
    16: "pcm_s16le",
    24: "pcm_s24le",
    32: "pcm_s32le",
}


def _ffmpeg_fingerprint() -> str:
    """blake3 of `ffmpeg -version` so the memo key tracks the converter version."""
    from smplstream.memo import tool_version_fingerprint

    return tool_version_fingerprint(["ffmpeg", "-version"])


def _build_ffmpeg_cmd(
    src: Path, dst: Path, *, fmt: str, sr: Optional[int], bits: Optional[int]
) -> list[str]:
    """Assemble the ffmpeg argv for the requested target. `dst` extension picks the muxer."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if sr is not None:
        cmd += ["-ar", str(sr)]
    if bits is not None:
        if fmt == "wav" and bits in _WAV_BITS_CODEC:
            cmd += ["-c:a", _WAV_BITS_CODEC[bits]]
        elif bits in _BITS_SAMPLE_FMT:
            cmd += ["-sample_fmt", _BITS_SAMPLE_FMT[bits]]
    cmd.append(str(dst))
    return cmd


def convert_audio_frame(
    audio_frame: dict,
    *,
    sr: Optional[int] = None,
    bits: Optional[int] = None,
    fmt: Optional[str] = None,
) -> list[dict]:
    """Convert one resolved `audio` frame, returning the new derived frame(s).

    Returns a single-element list with the converted `audio` frame on success, or a single
    `error` frame (code `op_failed` / `unsupported`) on failure — the caller passes inputs
    through first and appends whatever this returns.
    """
    from smplstream import cas, error_frame, frames as F

    in_id = audio_frame.get("id")
    in_hash = audio_frame.get("hash")
    in_meta = audio_frame.get("meta") or {}

    if not in_hash:
        return [error_frame("unsupported", "convert: input audio frame has no hash payload",
                            of=in_id, op=OP)]

    # Resolve target container: explicit --format wins; else keep the source container.
    target_fmt = (fmt or in_meta.get("fmt") or "wav").lower()
    if target_fmt not in _FORMAT_EXT:
        return [error_frame("unsupported", f"convert: unsupported format {target_fmt!r} "
                            f"(supported: {sorted(_FORMAT_EXT)})", of=in_id, op=OP)]
    ext, media = _FORMAT_EXT[target_fmt]

    if bits is not None and bits not in _BITS_SAMPLE_FMT:
        return [error_frame("unsupported", f"convert: unsupported bit depth {bits} "
                            f"(supported: {sorted(_BITS_SAMPLE_FMT)})", of=in_id, op=OP)]

    try:
        src = cas.get_path(in_hash)
    except FileNotFoundError as exc:
        return [error_frame("not_found", f"convert: {exc}", of=in_id, op=OP)]

    # Params capture the EXACT requested target so two spellings of the same request memoize
    # to one key; env_fingerprint binds the ffmpeg version (spec → Memoization).
    params: dict = {
        "format": target_fmt,
        "sr": sr,
        "bits": bits,
        "input_hash": in_hash,
        "env_fingerprint": _ffmpeg_fingerprint(),
    }

    with tempfile.TemporaryDirectory(prefix="smpl-convert-") as tmpd:
        dst = Path(tmpd) / f"converted.{ext}"
        cmd = _build_ffmpeg_cmd(src, dst, fmt=target_fmt, sr=sr, bits=bits)
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return [error_frame("op_failed", f"convert: ffmpeg failed to run: {exc}",
                                of=in_id, op=OP)]
        if proc.returncode != 0 or not dst.exists():
            msg = (proc.stderr or b"").decode("utf-8", "replace").strip() or "ffmpeg error"
            return [error_frame("op_failed", f"convert: ffmpeg exited {proc.returncode}: {msg}",
                                of=in_id, op=OP)]

        # NEW hash: cas.put_audio_file rehashes the converted PCM (native-rate/float32 canonical
        # of the *converted* output) — a different frame+hash from the source, by construction.
        out_hash = cas.put_audio_file(dst, media=media)

    out_meta = cas.read_meta(out_hash) or {}
    frame = F.audio_frame(
        out_hash,
        sr=out_meta.get("sr", sr or in_meta.get("sr", 0)),
        ch=out_meta.get("ch", in_meta.get("ch", 1)),
        dur=out_meta.get("dur", in_meta.get("dur", 0.0)),
        role="converted",
        of=in_id,
        lineage=[in_id] if in_id else None,
        op=OP,
        op_version=OP_VERSION,
        params=params,
        media=media,
        bits=bits,
        fmt=out_meta.get("fmt", target_fmt),
    )
    return [frame]
