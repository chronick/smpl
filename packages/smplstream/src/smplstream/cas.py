"""Content-addressed store (spec → *CAS*, *CAS integrity*, NORMATIVE).

- Location: ``~/.smpl/cas/`` (override ``SMPL_CAS_DIR``).
- Key: ``blake3:<hex>`` of canonical decoded PCM for audio, raw bytes for canonical blobs.
- Layout: sharded ``<aa>/<hex>.<ext>`` + sibling ``<hex>.meta.json`` for cheap metadata.
- Blobs immutable; atomic temp-write + ``rename``; a write whose recomputed hash != target
  is a fatal integrity error (bad bytes never land at the canonical path).
- Path safety: a hash must match ``^blake3:[0-9a-f]{64}$`` before it maps to a path, so a
  hostile ``blake3:../../etc/...`` can't traverse out (paths feed sox/ffmpeg/Demucs raw).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

from . import hashing
from .errors import IntegrityError, PathSafetyError

HASH_RE = re.compile(r"^blake3:[0-9a-f]{64}$")

# MIME → filesystem extension. The CAS stores the materialized bytes; ext is derived
# from the frame's `media` so external tools (sox/ffmpeg/Audacity) see a sane filename.
_MEDIA_EXT = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/aiff": "aiff",
    "audio/flac": "flac",
    "audio/mpeg": "mp3",
    "audio/midi": "mid",
    "image/png": "png",
    "image/jpeg": "jpg",
    "video/mp4": "mp4",
    "application/x-npy": "npy",
    "application/x-safetensors": "safetensors",
    "application/json": "json",
    "text/plain": "txt",
}


def cas_dir() -> Path:
    """Resolve the CAS root (reads ``SMPL_CAS_DIR`` each call so tests can override)."""
    return Path(os.environ.get("SMPL_CAS_DIR", "~/.smpl/cas")).expanduser()


def validate_hash(h: str) -> str:
    if not isinstance(h, str) or not HASH_RE.match(h):
        raise PathSafetyError(f"unsafe or malformed hash: {h!r}")
    return h


def _hex(h: str) -> str:
    return validate_hash(h).split(":", 1)[1]


def ext_for_media(media: Optional[str]) -> str:
    if not media:
        return "bin"
    return _MEDIA_EXT.get(media, "bin")


def _blob_path(h: str, ext: str) -> Path:
    hexd = _hex(h)
    return cas_dir() / hexd[:2] / f"{hexd}.{ext}"


def _meta_path(h: str) -> Path:
    hexd = _hex(h)
    return cas_dir() / hexd[:2] / f"{hexd}.meta.json"


def _atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=".tmp-", suffix=dest.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)  # atomic within the same directory
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def exists(h: str) -> bool:
    return _meta_path(h).exists()


def read_meta(h: str) -> Optional[dict]:
    mp = _meta_path(h)
    if not mp.exists():
        return None
    return json.loads(mp.read_text())


def get_path(h: str) -> Path:
    """Filesystem path to a stored blob (via its meta sidecar's recorded ext)."""
    meta = read_meta(h)
    if meta is None:
        raise FileNotFoundError(f"no CAS blob for {h}")
    path = _blob_path(h, meta.get("ext", "bin"))
    if not path.exists():
        raise FileNotFoundError(f"CAS meta present but blob missing for {h}")
    return path


def _write_meta(h: str, meta: dict) -> None:
    _atomic_write(_meta_path(h), json.dumps(meta, sort_keys=True).encode("utf-8"))


def put_blob(data: bytes, media: str, *, expected_hash: Optional[str] = None) -> str:
    """Store a canonical blob (PNG, .npy, JSON, text). Key = blake3 over the bytes."""
    h = hashing.blob_hash(data)
    if expected_hash is not None and expected_hash != h:
        raise IntegrityError(f"blob hash {h} != expected {expected_hash}")
    ext = ext_for_media(media)
    if not exists(h):
        _atomic_write(_blob_path(h, ext), data)
        _write_meta(h, {"hash": h, "media": media, "ext": ext, "size": len(data)})
    return h


def put_audio_bytes(
    wav_bytes: bytes, *, media: str = "audio/wav", expected_hash: Optional[str] = None
) -> str:
    """Store encoded audio bytes, keyed by the *canonical decoded-PCM* hash.

    The stored bytes are the source encoding (faithful round-trip on ``resolve``), but the
    key is the canonical-PCM hash so two encodings that decode bit-identically dedup, and
    re-encodes that change the PCM correctly get a new key.
    """
    pcm, sr, ch = hashing.decode_canonical_bytes(wav_bytes)
    h = hashing.audio_hash_from_pcm(pcm, sr, ch)
    if expected_hash is not None and expected_hash != h:
        raise IntegrityError(f"audio hash {h} != expected {expected_hash}")
    ext = ext_for_media(media)
    if not exists(h):
        probe = hashing.probe_audio(wav_bytes)
        _atomic_write(_blob_path(h, ext), wav_bytes)
        _write_meta(
            h,
            {
                "hash": h,
                "media": media,
                "ext": ext,
                "size": len(wav_bytes),
                "sr": sr,
                "ch": ch,
                "dur": probe["dur"],
                "fmt": probe["fmt"],
                "subtype": probe["subtype"],
            },
        )
    return h


def put_audio_file(
    path: str | Path, *, media: str = "audio/wav", expected_hash: Optional[str] = None
) -> str:
    """Store an audio file by its canonical-PCM hash (stores the source bytes verbatim)."""
    raw = Path(path).read_bytes()
    # Derive media/ext from the source extension when it is a known audio type.
    src_ext = Path(path).suffix.lower().lstrip(".")
    for m, e in _MEDIA_EXT.items():
        if e == src_ext and m.startswith("audio/"):
            media = m
            break
    return put_audio_bytes(raw, media=media, expected_hash=expected_hash)


# ---------------------------------------------------------------------------
# Garbage collection (spec → *CAS integrity / GC safety*).
# v1 ships the SAFETY rule (never delete a live/in-flight blob); GC *policy* (TTL,
# thresholds) is intentionally minimal and conservative.
# ---------------------------------------------------------------------------


def _lock_path() -> Path:
    return cas_dir() / ".gc.lock"


def iter_blobs():
    """Yield ``(hash, blob_path, meta_path)`` for every stored blob."""
    root = cas_dir()
    if not root.exists():
        return
    for shard in sorted(root.iterdir()):
        if not shard.is_dir() or len(shard.name) != 2:
            continue
        for meta in sorted(shard.glob("*.meta.json")):
            hexd = meta.name[: -len(".meta.json")]
            h = "blake3:" + hexd
            try:
                ext = json.loads(meta.read_text()).get("ext", "bin")
            except (OSError, json.JSONDecodeError):
                continue
            yield h, shard / f"{hexd}.{ext}", meta


def gc(
    *,
    keep: Optional[set[str]] = None,
    grace_seconds: float = 3600.0,
    dry_run: bool = True,
) -> dict:
    """Conservatively collect unreferenced blobs.

    Never deletes a blob in ``keep`` (referenced by live frames), nor one whose mtime is
    within ``grace_seconds`` (in-flight / recently produced). Holds an exclusive lock so a
    concurrent producer can't race a delete. Returns a summary; ``dry_run`` reports only.
    """
    keep = keep or set()
    root = cas_dir()
    root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    removed, kept, reserved = [], [], []

    lock = _lock_path()
    # Best-effort exclusive lock via O_EXCL create; refuse to GC if one is held.
    try:
        lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise IntegrityError("CAS GC lock held by another process; refusing to collect")
    try:
        for h, blob_path, meta_path in iter_blobs():
            if h in keep:
                kept.append(h)
                continue
            try:
                age = now - blob_path.stat().st_mtime
            except OSError:
                continue
            if age < grace_seconds:
                reserved.append(h)  # in grace window → in-flight, never delete
                continue
            if not dry_run:
                for p in (blob_path, meta_path):
                    try:
                        p.unlink()
                    except OSError:
                        pass
            removed.append(h)
    finally:
        os.close(lock_fd)
        try:
            os.unlink(lock)
        except OSError:
            pass

    return {
        "removed": removed,
        "kept": kept,
        "reserved_in_grace": reserved,
        "dry_run": dry_run,
        "grace_seconds": grace_seconds,
    }
