"""CAS integrity: atomicity, dedup, path-safety, faithful round-trip, GC safety."""

from __future__ import annotations

import pytest

from smplstream import cas, hashing
from smplstream.errors import IntegrityError, PathSafetyError


def test_put_and_resolve_roundtrip(isolated_cas, tone_wav_bytes):
    wav = tone_wav_bytes()
    h = cas.put_audio_bytes(wav)
    assert h.startswith("blake3:")
    assert cas.exists(h)
    path = cas.get_path(h)
    assert path.exists()
    assert path.read_bytes() == wav  # faithful: stored source bytes verbatim


def test_dedup_same_pcm(isolated_cas, tone_wav_bytes):
    wav = tone_wav_bytes()
    h1 = cas.put_audio_bytes(wav)
    h2 = cas.put_audio_bytes(wav)
    assert h1 == h2
    blobs = list(cas.iter_blobs())
    assert len(blobs) == 1


def test_integrity_check_rejects_wrong_expected(isolated_cas, tone_wav_bytes):
    wav = tone_wav_bytes()
    with pytest.raises(IntegrityError):
        cas.put_audio_bytes(wav, expected_hash="blake3:" + "0" * 64)


def test_path_safety_rejects_traversal(isolated_cas):
    for bad in ("blake3:../../etc/passwd", "blake3:" + "g" * 64, "notahash", "blake3:abc"):
        with pytest.raises(PathSafetyError):
            cas.validate_hash(bad)


def test_blob_put_dedup_and_meta(isolated_cas):
    data = b"\x89PNG fake bytes"
    h = cas.put_blob(data, "image/png")
    assert cas.read_meta(h)["ext"] == "png"
    assert cas.get_path(h).read_bytes() == data


def test_gc_keeps_referenced_and_respects_grace(isolated_cas, tone_wav_bytes):
    h = cas.put_audio_bytes(tone_wav_bytes())
    # Fresh blob is within the grace window → reserved, never deleted.
    summary = cas.gc(keep=set(), grace_seconds=3600, dry_run=False)
    assert h in summary["reserved_in_grace"]
    assert cas.exists(h)
    # With zero grace it becomes eligible, but `keep` protects it.
    summary = cas.gc(keep={h}, grace_seconds=0, dry_run=False)
    assert h in summary["kept"]
    assert cas.exists(h)
    # Zero grace, not kept → collected.
    summary = cas.gc(keep=set(), grace_seconds=0, dry_run=False)
    assert h in summary["removed"]
    assert not cas.exists(h)


def test_gc_dry_run_does_not_delete(isolated_cas, tone_wav_bytes):
    h = cas.put_audio_bytes(tone_wav_bytes())
    summary = cas.gc(keep=set(), grace_seconds=0, dry_run=True)
    assert h in summary["removed"]
    assert cas.exists(h)  # dry-run: still there


def test_gc_reclaims_orphan_blob(isolated_cas):
    # Simulate a crash between the blob write and the meta write: a blob with no sidecar.
    h = "blake3:" + "ab" * 32
    cas._atomic_write(cas._blob_path(h, "wav"), b"orphan bytes")
    assert not cas.exists(h)  # no meta → not "present" via exists()
    found = [bh for bh, _, mp in cas.iter_blobs() if mp is None]
    assert h in found  # iter_blobs surfaces the orphan
    summary = cas.gc(keep=set(), grace_seconds=0, dry_run=False)
    assert h in summary["removed"]
    assert not cas._blob_path(h, "wav").exists()  # orphan reclaimed


def test_ext_revalidated_on_read(isolated_cas):
    from smplstream.errors import PathSafetyError

    with pytest.raises(PathSafetyError):
        cas._blob_path("blake3:" + "ab" * 32, "wav/../../etc")  # crafted ext rejected
