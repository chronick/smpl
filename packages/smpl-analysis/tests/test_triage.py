"""Tests for smpl_analysis.triage — batch-triage viz layer (ticket vault-22oy).

Covers the deterministic scoring core (robust-z + verdict bands), the contact-sheet
ranking-validation acceptance (one Read → correct ranking), and the spectrum-vs-profile
overlay rendering for any candidate + stats-file pair.

A/B FINDING (tile-budget sanity, per the acceptance):
    The batch RANKING is fully carried by the deterministic scalar layer — `rank_candidates`
    (and the `contact-sheet-index` text frame it emits) produce the exact best→worst order
    with ZERO image tokens. `test_index_ranking_matches_image_order` proves the text index
    and the composite PNG agree. So "mel-only + a scalar strip" triages the *ordering* just
    as well as multi-image tiles — the composite PNG earns its token cost only for the
    qualitative read (gestalt / anomaly-spotting on uncertain survivors), which is why the
    sheet is emitted as ONE composite (not N singles) and per-candidate detail overlays are
    produced lazily via `profile-overlay`.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pytest
import soundfile as sf

from smplstream import cas, frames as F

from smpl_analysis import octave as OCT, stats as ST, triage as T

SR = 44100


def _is_png(b: bytes) -> bool:
    return b[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def cas_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.setenv("SMPL_STATS_DIR", str(tmp_path / "stats"))
    return tmp_path


def _audio(freq, name, sr=SR, dur=0.4):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = (0.4 * np.sin(2 * np.pi * freq * t)).astype("float32").reshape(-1, 1)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    af = F.audio_frame(h, sr=meta.get("sr", sr), ch=meta.get("ch", 1),
                       dur=meta.get("dur", dur), role="source")
    af["meta"]["name"] = name  # meta is not part of the content id → safe post-mint
    return af


def _metric_feat(audio_id, value):
    return F.feature_frame({"metric": float(value)}, role="metric", of=audio_id)


def _noise(rng, sr=SR, dur=0.5):
    return (0.2 * rng.standard_normal(int(sr * dur))).astype("float32")


# --- scoring core ----------------------------------------------------------------------------


def test_robust_z_linear_domain():
    stat = {"median": 0.0, "mad": 2.0, "domain": "linear"}
    assert T.robust_z(4.0, stat) == pytest.approx(2.0)
    assert T.robust_z(-3.0, stat) == pytest.approx(-1.5)


def test_robust_z_log_domain():
    # median in log10 space is 2.0 (i.e. 100); a candidate of 1000 → log10=3 → z=(3-2)/0.5=2
    stat = {"median": 2.0, "mad": 0.5, "domain": "log10"}
    assert T.robust_z(1000.0, stat) == pytest.approx(2.0)
    assert T.robust_z(-1.0, stat) is None          # non-positive in a log key is unscorable


def test_verdict_bands():
    assert T.axis_band(1.0) == "in_band"
    assert T.axis_band(2.5) == "above"
    assert T.axis_band(-2.5) == "below"
    assert T.axis_band(4.0) == "far_above"
    assert T.axis_band(-4.0) == "far_below"
    assert T.verdict_word(1.0) == "PASS"
    assert T.verdict_word(2.5) == "ALTER"
    assert T.verdict_word(4.0) == "REJECT"


def test_score_candidate_aggregates_and_orders_worst():
    profile = {
        "a": {"median": 0.0, "mad": 1.0, "domain": "linear"},
        "b": {"median": 0.0, "mad": 1.0, "domain": "linear"},
    }
    sc = T.score_candidate({"a": 1.0, "b": 4.0}, profile)
    assert sc["n_keys"] == 2
    assert sc["agg"] == pytest.approx(2.5)          # mean(|1|, |4|)
    assert sc["max_abs"] == pytest.approx(4.0)
    assert sc["worst"][0][0] == "b"                 # worst axis first


def test_score_candidate_no_overlap_sorts_last():
    sc = T.score_candidate({"other": 1.0}, {"a": {"median": 0.0, "mad": 1.0, "domain": "linear"}})
    assert sc["n_keys"] == 0
    assert sc["agg"] == float("inf")


# --- contact-sheet ranking acceptance --------------------------------------------------------

# A single linear key, median 0 / MAD 1 → agg |z| == |value|. Deviations are chosen distinct.
_PROFILE = {"metric": {"median": 0.0, "mad": 1.0, "p10": -1.0, "p50": 0.0, "p90": 1.0,
                       "n": 30, "domain": "linear"}}
_CANDIDATES = [
    ("c-freq100", 100.0, 0.1),
    ("c-freq150", 150.0, -0.4),
    ("c-freq200", 200.0, 0.9),
    ("c-freq250", 250.0, 2.3),
    ("c-freq300", 300.0, -1.7),
    ("c-freq350", 350.0, 3.9),
    ("c-freq400", 400.0, 0.5),
    ("c-freq450", 450.0, -2.8),
]


def _batch(cas_dir):
    frames = []
    for name, freq, value in _CANDIDATES:
        af = _audio(freq, name)
        frames.append(af)
        frames.append(_metric_feat(af["id"], value))
    return frames


def _expected_order():
    # Independent scalar ranking: ascending |value| (== agg |z|), tie-broken by label.
    return [name for name, _f, _v in sorted(_CANDIDATES, key=lambda c: (abs(c[2]), c[0]))]


def test_rank_candidates_matches_scalar_ranking(cas_dir):
    ranked = T.rank_candidates(_batch(cas_dir), _PROFILE)
    assert [e["label"] for e in ranked] == _expected_order()


def test_contact_sheet_tile_order_matches_scalar_ranking(cas_dir):
    """Acceptance: one Read of the sheet lets the agent rank the 8-candidate batch."""
    out = T.render_contact_sheet(_batch(cas_dir), _PROFILE)
    image = next(f for f in out if f.get("role") == "contact-sheet")
    # the burned-in / recorded tile order IS the scalar z-ranking — the model must not re-sort
    assert image["params"]["order"] == _expected_order()
    assert image["params"]["n"] == 8
    # the composite PNG actually rendered and landed in the CAS
    assert _is_png(cas.get_path(image["hash"]).read_bytes())
    assert F.validate_frame(image) == []


def test_index_ranking_matches_image_order(cas_dir):
    """The scalar-strip text index and the composite PNG agree (the A/B basis)."""
    out = T.render_contact_sheet(_batch(cas_dir), _PROFILE)
    image = next(f for f in out if f.get("role") == "contact-sheet")
    index = next(f for f in out if f.get("role") == "contact-sheet-index")
    assert index["params"]["order"] == image["params"]["order"] == _expected_order()
    # the index is legible text (a markdown table) — zero image tokens to read the ranking
    assert "candidate" in index["data"] and "verdict" in index["data"]


def test_contact_sheet_verdict_bands(cas_dir):
    out = T.render_contact_sheet(_batch(cas_dir), _PROFILE)
    image = next(f for f in out if f.get("role") == "contact-sheet")
    by_label = {s["label"]: s for s in image["params"]["scores"]}
    assert by_label["c-freq100"]["verdict"] == "PASS"     # |z|=0.1
    assert by_label["c-freq250"]["verdict"] == "ALTER"    # |z|=2.3
    assert by_label["c-freq350"]["verdict"] == "REJECT"   # |z|=3.9


def test_contact_sheet_empty_batch_is_safe(cas_dir):
    assert T.render_contact_sheet([], _PROFILE) == []


def test_contact_sheet_short_candidate_renders_tile(cas_dir):
    """Regression (vault-2t9g): a sub-window candidate (single-cycle sub, tiny one-shot)
    still gets a real mel tile instead of a dropped/'no audio' tile, and never trips
    librosa's 'n_fft too large' path."""
    frames = []
    for name, freq, value in _CANDIDATES[:2]:
        af = _audio(freq, name)
        frames.append(af)
        frames.append(_metric_feat(af["id"], value))
    short = _audio(200.0, "c-short", dur=120 / SR)  # 120 samples, far under the 2048 window
    frames.append(short)
    frames.append(_metric_feat(short["id"], 0.2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = T.render_contact_sheet(frames, _PROFILE)
    image = next(f for f in out if f.get("role") == "contact-sheet")
    assert image["params"]["n"] == 3
    assert "c-short" in image["params"]["order"]
    assert _is_png(cas.get_path(image["hash"]).read_bytes())
    assert not any("too large for input signal" in str(w.message) for w in caught)


# --- spectrum-vs-profile overlay -------------------------------------------------------------


def test_overlay_renders_with_real_spectrum_profile(cas_dir):
    """Build a real spectrum profile via octave-spectrum → stats build, then overlay a candidate."""
    rng = np.random.default_rng(7)
    corpus = []
    for _ in range(4):
        af = _audio_from_signal(_noise(rng))
        corpus.append(af)
        corpus.extend(OCT.octave_audio_frame(af))
    ST.build_stats(corpus, role="pad")
    profile = T.load_stats_file(str(cas_dir / "stats" / "pad.json"))
    assert any(k.startswith("spectrum.oct6.") for k in profile)

    candidate = _audio_from_signal(_noise(rng))
    out = T.render_profile_overlay(candidate, profile)
    assert len(out) == 1
    img = out[0]
    assert img["kind"] == "image" and img["role"] == "profile-overlay"
    assert img["params"]["has_profile"] is True
    assert set(img["params"]["band_deltas"]) >= {"sub", "bass", "mid", "air"}
    assert _is_png(cas.get_path(img["hash"]).read_bytes())
    assert F.validate_frame(img) == []


def test_overlay_renders_without_spectrum_profile(cas_dir):
    """Any candidate + stats-file pair renders a valid PNG, even with no spectrum keys."""
    candidate = _audio_from_signal(_noise(np.random.default_rng(3)))
    profile = {"envelope.attack_ms_10_90": {"median": 0.3, "mad": 0.1, "domain": "log10"}}
    out = T.render_profile_overlay(candidate, profile)
    img = out[0]
    assert img["params"]["has_profile"] is False
    assert _is_png(cas.get_path(img["hash"]).read_bytes())


def _audio_from_signal(sig, sr=SR):
    if sig.ndim == 1:
        sig = sig.reshape(-1, 1)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    return F.audio_frame(h, sr=meta.get("sr", sr), ch=meta.get("ch", 1),
                         dur=meta.get("dur", sig.shape[0] / sr), role="source")
