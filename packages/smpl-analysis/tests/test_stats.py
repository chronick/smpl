"""Tests for role-corpus statistics (smpl_analysis.stats; ticket vault-3tuy).

Builds corpora of synthesized `feature` frames and asserts the reduction carries
{median, MAD, p10, p50, p90, n} per key in the log domain, is generic (no role/threshold
baked in), writes the role JSON, and memo-caches on (corpus content set, op_version).
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import stats as ST

SR = 44100

_STAT_KEYS = {"median", "mad", "p10", "p50", "p90", "n", "domain"}


def _feat(data, of="a0"):
    """A minimal feature frame carrying a data dict (mirrors the on-wire shape)."""
    from smplstream import frames as F

    return F.feature_frame(dict(data), role="envelope", of=of)


def _audio_frame(sig, sr=SR):
    from smplstream import cas, frames as F

    if sig.ndim == 1:
        sig = sig.reshape(-1, 1)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    return F.audio_frame(h, sr=meta.get("sr", sr), ch=meta.get("ch", 1),
                         dur=meta.get("dur", sig.shape[0] / sr), role="source")


# --- reduction -------------------------------------------------------------------------------


def test_reduce_positive_key_uses_log_domain():
    r = ST._reduce([10.0, 100.0, 1000.0])
    assert r["domain"] == "log10"
    assert r["n"] == 3
    # log10 of [1,2,3] → median 2.0
    assert r["median"] == pytest.approx(2.0, abs=1e-6)
    assert _STAT_KEYS <= set(r)


def test_reduce_signed_key_stays_linear():
    r = ST._reduce([-0.2, -0.1, 0.0, -0.15])   # a slope that can go non-positive
    assert r["domain"] == "linear"
    assert r["n"] == 4


def test_reduce_empty_is_safe():
    r = ST._reduce([])
    assert r["n"] == 0 and r["median"] is None


def test_scalar_of_handles_shapes():
    assert ST._scalar_of(3.5) == 3.5
    assert ST._scalar_of({"mean": 2.0, "stdev": 1.0}) == 2.0
    assert ST._scalar_of({"median": 7.0}) == 7.0
    assert ST._scalar_of(True) is None          # booleans are not distributions
    assert ST._scalar_of("x") is None


# --- collect + build -------------------------------------------------------------------------


def _corpus(n=8):
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n):
        frames.append(_feat({
            "envelope.attack_ms_10_90": float(2.0 + rng.normal(0, 0.2)),
            "envelope.sustain_ratio_150ms": float(0.1 + rng.normal(0, 0.02)),
            "envelope.early_decay_slope": float(-0.12 + rng.normal(0, 0.01)),
        }, of=f"a{i}"))
    return frames


def test_build_stats_reduces_every_key_with_n(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.setenv("SMPL_STATS_DIR", str(tmp_path / "stats"))
    frames = _corpus(8)
    derived = ST.build_stats(frames, role="kick")
    assert len(derived) == 1
    frame = derived[0]
    assert frame["kind"] == "feature"
    assert frame["role"] == "stats:kick"
    assert frame["op_version"] == "stats@1"
    data = frame["data"]
    for key in ("envelope.attack_ms_10_90", "envelope.sustain_ratio_150ms",
                "envelope.early_decay_slope"):
        assert key in data
        assert _STAT_KEYS <= set(data[key])
        assert data[key]["n"] == 8               # MUST carry n
    # positive keys reduced in log domain, the signed slope stays linear
    assert data["envelope.attack_ms_10_90"]["domain"] == "log10"
    assert data["envelope.early_decay_slope"]["domain"] == "linear"
    # corpus n is carried on the frame params too
    assert frame["params"]["n"] == 8
    assert frame["params"]["role"] == "kick"


def test_build_stats_writes_role_json(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.setenv("SMPL_STATS_DIR", str(tmp_path / "stats"))
    ST.build_stats(_corpus(5), role="pad")
    role_file = tmp_path / "stats" / "pad.json"
    assert role_file.exists()
    doc = json.loads(role_file.read_text())
    assert doc["role"] == "pad"
    assert doc["n"] == 5
    assert "envelope.attack_ms_10_90" in doc["stats"]
    assert "cas_hash" in doc


def test_build_stats_is_memo_cached_on_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.setenv("SMPL_STATS_DIR", str(tmp_path / "stats"))
    frames = _corpus(6)
    first = ST.build_stats(frames, role="kick")[0]
    second = ST.build_stats(frames, role="kick")[0]
    assert first["params"]["cache_hit"] is False
    assert second["params"]["cache_hit"] is True
    assert first["params"]["cas_hash"] == second["params"]["cas_hash"]
    assert first["params"]["memo_key"] == second["params"]["memo_key"]


def test_stats_generic_over_arbitrary_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.setenv("SMPL_STATS_DIR", str(tmp_path / "stats"))
    # arbitrary namespaced keys (incl. a {mean,stdev} stat object) → still reduced
    frames = [
        _feat({"lowlevel.spectral_rolloff": {"mean": 5000.0 + i * 100, "stdev": 10.0},
               "custom.metric": float(i + 1)}, of=f"a{i}")
        for i in range(4)
    ]
    data = ST.build_stats(frames, role="whatever")[0]["data"]
    assert "lowlevel.spectral_rolloff" in data
    assert "custom.metric" in data
    assert data["custom.metric"]["n"] == 4


def test_corpus_content_set_prefers_audio_hashes(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _audio_frame((0.3 * np.sin(2 * np.pi * 100 * np.arange(SR) / SR)).astype("float32"))
    cset = ST.corpus_content_set([af, _feat({"x": 1.0})])
    assert cset == [af["hash"]]


def test_op_version_constant():
    assert ST.OP == "stats"
    assert ST.OP_VERSION == "stats@1"
