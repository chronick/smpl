"""Selection semantics + root-cause error propagation (spec → *Selection*, *Error model*)."""

from __future__ import annotations

import pytest

from smplstream import select as S
from smplstream import error_frame
from smplstream.errors import ResolutionError


def _audio(role, fid, seq=None):
    fr = {"v": 1, "kind": "audio", "id": fid, "role": role,
          "hash": "blake3:" + "a" * 64, "media": "audio/wav", "meta": {"sr": 48000, "ch": 1, "dur": 1.0}}
    if seq is not None:
        fr["seq"] = seq
    return fr


def test_last_wins():
    frames = [_audio("stem:drums", "id1"), _audio("stem:drums", "id2")]
    assert S.select(frames, role="stem:drums")[0]["id"] == "id2"


def test_all_returns_every_match():
    frames = [_audio("stem:drums", "id1"), _audio("stem:drums", "id2")]
    assert len(S.select(frames, role="stem:drums", mode="all")) == 2


def test_strict_errors_on_multiple():
    frames = [_audio("stem:drums", "id1"), _audio("stem:drums", "id2")]
    with pytest.raises(ResolutionError):
        S.select(frames, role="stem:drums", mode="strict")


def test_seq_tiebreak():
    frames = [_audio("x", "id1", seq=5), _audio("x", "id2", seq=2)]
    # last-wins by seq, not stream position → id1 (seq 5) wins.
    assert S.select(frames, role="x")[0]["id"] == "id1"


def test_resolve_single_audio_not_found():
    with pytest.raises(ResolutionError) as ei:
        S.resolve_single_audio([], role="stem:bass")
    assert ei.value.code == "not_found"


def test_root_cause_error_propagates():
    # stems failed for the bass: a downstream resolve must surface the OOM, not "not found".
    err = error_frame("resource_exhausted", "CUDA OOM in demucs", of="blake3:" + "c" * 64, op="stems")
    # Make a role-bearing frame whose ancestor is the failed id.
    child = {"v": 1, "kind": "audio", "id": "child", "role": "stem:bass",
             "lineage": ["blake3:" + "c" * 64]}
    # No payload for the role (child has no hash) → must raise the root cause.
    child.pop("hash", None)
    with pytest.raises(ResolutionError) as ei:
        S.resolve_single_audio([err, child], role="stem:bass")
    assert ei.value.code == "resource_exhausted"
