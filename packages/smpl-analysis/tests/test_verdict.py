"""Tests for smpl_analysis.verdict — the gen-QC judge op (ticket vault-1apl).

Covers the routing + confidence logic (keep/listen/cut/alter), the axes[] discretization, the
three-band listen downgrade on a thin corpus, and that an emitted frame conforms to
verdict-frames.md (passes `validate_frame`). The judge reuses triage's ONE scoring core; these
tests exercise the discretization + routing layered on top.

Synthesized profiles use a single linear key (median 0 / MAD 1 → z == value) so the routing
thresholds are exact and legible.
"""

from __future__ import annotations

from smplstream import frames as F, validate_frame

from smpl_analysis import verdict as V


def _profile(n=30, keys=("a", "b")):
    return {k: {"median": 0.0, "mad": 1.0, "p10": -1.0, "p50": 0.0, "p90": 1.0,
                "n": n, "domain": "linear"} for k in keys}


def _meta(n=30, role="kick"):
    return {"role": role, "n": n, "corpus_size": n, "cas_hash": "blake3:" + "c" * 64,
            "op_version": "stats@1"}


# --- (a) a candidate matching the profile → keep, all axes in_band ----------------------------


def test_match_profile_keeps_all_in_band():
    data = V.judge({"a": 0.2, "b": -0.3}, _profile(), _meta())
    assert data["rollup"]["decision"] == "keep"
    assert data["rollup"]["confidence"] == "high"
    assert all(ax["verdict"] == "in_band" for ax in data["axes"])
    # ref carries median/mad/n; axes are worst-first.
    assert data["axes"][0]["ref"] == {"median": 0.0, "mad": 1.0, "n": 30}
    assert abs(data["axes"][0]["z"]) >= abs(data["axes"][1]["z"])


# --- (b) far off on one axis → that axis far_*, decision cut, dominant_axis correct -----------


def test_far_off_one_axis_cuts_and_names_dominant():
    data = V.judge({"a": 0.1, "b": 5.0}, _profile(), _meta())
    assert data["rollup"]["decision"] == "cut"
    assert data["rollup"]["confidence"] == "high"
    assert data["rollup"]["dominant_axis"] == "b"
    by_key = {ax["key"]: ax for ax in data["axes"]}
    assert by_key["b"]["verdict"] == "far_above"
    assert by_key["a"]["verdict"] == "in_band"


def test_far_off_thin_corpus_downgrades_to_listen():
    # Same hard fail, but a thin corpus (n<12) never auto-cuts → listen.
    data = V.judge({"a": 0.1, "b": 5.0}, _profile(n=5), _meta(n=5))
    assert data["rollup"]["decision"] == "listen"
    assert data["rollup"]["confidence"] == "low"
    assert data["rollup"]["reason"] == "insufficient_corpus"
    assert data["rollup"]["dominant_axis"] == "b"


def test_middle_band_is_alter():
    # 2.0 <= max_abs < 3.5 → the repairable middle band.
    data = V.judge({"a": 0.1, "b": 2.6}, _profile(), _meta())
    assert data["rollup"]["decision"] == "alter"
    assert data["rollup"]["confidence"] == "med"  # boundary-ish, never "high"


# --- (c) thin corpus → confidence low / routed to listen -------------------------------------


def test_thin_corpus_keep_becomes_listen():
    data = V.judge({"a": 0.2, "b": -0.1}, _profile(n=5), _meta(n=5))
    assert data["rollup"]["confidence"] == "low"
    assert data["rollup"]["decision"] == "listen"  # a would-be keep, but evidence is thin
    assert data["rollup"]["reason"] == "insufficient_corpus"


def test_no_overlap_is_low_and_listen():
    data = V.judge({"unrelated": 1.0}, _profile(), _meta())
    assert data["rollup"]["confidence"] == "low"
    assert data["rollup"]["reason"] == "no_overlap"
    assert data["rollup"]["decision"] == "listen"
    assert data["axes"] == []
    assert data["rollup"]["distance"] is None


# --- (d) emitted frame conforms (validate_frame) ---------------------------------------------


def test_emitted_verdict_frame_validates():
    audio = F.audio_frame("blake3:" + "a" * 64, sr=48000, ch=1, dur=0.5, role="source")
    feats = [
        F.feature_frame({"a": 0.2}, role="metric", of=audio["id"]),
        F.feature_frame({"b": 5.0}, role="metric", of=audio["id"]),
    ]
    stream = [audio, *feats]
    out = V.verdict_frames(stream, _profile(), _meta())
    assert len(out) == 1
    v = out[0]
    assert validate_frame(v) == []
    assert v["kind"] == "verdict"
    assert v["of"] == audio["id"]
    assert v["schema_version"]  # required, non-empty
    # lineage = the candidate's feature frame ids + the stats artifact.
    assert set(f["id"] for f in feats).issubset(set(v["lineage"]))
    assert v["data"]["rollup"]["decision"] == "cut"
    assert v["data"]["rollup"]["dominant_axis"] == "b"


def test_target_brief_adds_in_tolerance():
    # With a --target brief, axes carry in_tolerance vs the absolute band; omitted otherwise.
    data_no_target = V.judge({"a": 0.2, "b": -0.3}, _profile(), _meta())
    assert "in_tolerance" not in data_no_target["axes"][0]

    data = V.judge({"a": 0.2, "b": -0.3}, _profile(), _meta(),
                   target={"a": (-1.0, 1.0), "b": (0.0, 1.0)})
    by_key = {ax["key"]: ax for ax in data["axes"]}
    assert by_key["a"]["in_tolerance"] is True    # 0.2 in [-1, 1]
    assert by_key["b"]["in_tolerance"] is False   # -0.3 not in [0, 1]
    assert data["target"] == "brief:absolute-tolerances"
