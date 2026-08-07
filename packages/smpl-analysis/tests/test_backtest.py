"""Tests for smpl_analysis.backtest — the verdict calibration harness (ticket vault-2kyt).

Runs the gate over the SEED corpus (tests/fixtures/verdict_backtest_seed.jsonl, scored against
verdict_backtest_seed_profile.json) whose every outcome is hand-verified, and asserts the
confusion matrix, precision/recall, headline auto-keep accuracy, and the threshold-sensitivity
sweep are computed correctly. The seed is engineered so auto-keep accuracy = 0.75 — deliberately
BELOW the 0.85 target — because one entry (`keep-char-fail`) is band-clean but wrong in character
(the residual "too happy / synthwave" failure only a character axis / CLAP catches). The harness
exists to surface exactly that.

A drift guard pins backtest.decide (the parametric router used for the sensitivity sweep) to the
real gate (verdict.judge) at the baseline thresholds.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import backtest as B
from smpl_analysis import triage as T
from smpl_analysis import verdict as V

FIXTURES = Path(__file__).parent / "fixtures"
SEED_CORPUS = FIXTURES / "verdict_backtest_seed.jsonl"
SEED_PROFILE = FIXTURES / "verdict_backtest_seed_profile.json"


def _load():
    corpus = B.load_corpus(str(SEED_CORPUS))
    profile, meta = V.load_profile(str(SEED_PROFILE))
    return corpus, {"profile": profile, "meta": meta}


def _report():
    corpus, profiles = _load()
    return B.run_backtest(corpus, profiles, corpus_path=str(SEED_CORPUS))


# --- corpus loading --------------------------------------------------------------------------


def test_load_corpus_skips_comments_and_blanks():
    corpus = B.load_corpus(str(SEED_CORPUS))
    assert len(corpus) == 10
    assert corpus[0]["id"] == "keep-01"
    assert all("human_verdict" in e for e in corpus)


def test_load_corpus_accepts_json_array(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('[{"id":"x","role":"kick","features":{"a":0.1},"human_verdict":"keep"}]')
    corpus = B.load_corpus(str(p))
    assert corpus == [{"id": "x", "role": "kick", "features": {"a": 0.1}, "human_verdict": "keep"}]


# --- fold + parametric decision --------------------------------------------------------------


def test_fold3_folds_alter_to_listen():
    assert B.fold3("alter") == "listen"
    assert B.fold3("keep") == "keep"
    assert B.fold3("cut") == "cut"
    assert B.fold3("listen") == "listen"


def test_decide_mirrors_the_real_gate_at_baseline():
    # Drift guard: backtest.decide at (Z_IN_BAND, Z_FAR) MUST equal verdict.judge's raw decision.
    corpus, profiles = _load()
    scored, _ = B.score_entries(corpus, profiles)
    assert len(scored) == 10
    for e in scored:
        d, _conf = B.decide(e["_max_abs"], e["min_n"], e["n_keys"],
                            z_in_band=T.Z_IN_BAND, z_far=T.Z_FAR)
        assert d == e["gate_raw"], f"{e['id']}: decide={d} gate={e['gate_raw']}"


# --- confusion matrix + headline numbers (all hand-verified) ---------------------------------


def test_confusion_matrix_exact():
    r = _report()
    cm = r["confusion"]
    assert cm["keep"] == {"keep": 3, "listen": 1, "cut": 0}
    assert cm["listen"] == {"keep": 0, "listen": 1, "cut": 1}
    assert cm["cut"] == {"keep": 1, "listen": 1, "cut": 2}
    assert r["n_scored"] == 10 and r["n_skipped"] == 0


def test_auto_keep_accuracy_is_below_target():
    r = _report()
    # 3 of the gate's 4 auto-keeps were human keeps → 0.75; keep-char-fail is the false accept.
    assert r["auto_keep_accuracy"] == pytest.approx(0.75)
    assert r["target_auto_keep"] == 0.85
    assert r["meets_target"] is False
    assert r["auto_cut_accuracy"] == pytest.approx(2 / 3)
    assert r["overall_accuracy"] == pytest.approx(0.6)


def test_per_class_precision_recall():
    pc = _report()["per_class"]
    assert pc["keep"]["precision"] == pytest.approx(0.75)
    assert pc["keep"]["recall"] == pytest.approx(0.75)
    assert pc["keep"]["support_human"] == 4 and pc["keep"]["support_gate"] == 4
    assert pc["listen"]["precision"] == pytest.approx(1 / 3)
    assert pc["listen"]["recall"] == pytest.approx(0.5)
    assert pc["cut"]["precision"] == pytest.approx(2 / 3)
    assert pc["cut"]["recall"] == pytest.approx(0.5)
    assert pc["cut"]["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))


# --- threshold sensitivity -------------------------------------------------------------------


def test_threshold_sensitivity_auto_keep_vs_z_in_band():
    sens = _report()["threshold_sensitivity"]["z_in_band"]
    by_val = {r["value"]: r for r in sens}
    # Tightening the keep band doesn't recover the character false-accept (it sits at z=0.3);
    # widening it pulls a `listen` into keep and DROPS accuracy — the harness's real finding.
    assert by_val[1.5]["auto_keep_accuracy"] == pytest.approx(0.75)
    assert by_val[1.5]["n_gate_keep"] == 4
    assert by_val[2.0]["auto_keep_accuracy"] == pytest.approx(0.75)
    assert by_val[2.0]["baseline"] is True
    assert by_val[2.5]["auto_keep_accuracy"] == pytest.approx(0.6)
    assert by_val[2.5]["n_gate_keep"] == 5


def test_threshold_sensitivity_auto_cut_vs_z_far():
    sens = _report()["threshold_sensitivity"]["z_far"]
    by_val = {r["value"]: r for r in sens}
    # Lowering Z_FAR to 3.0 auto-culls alter-03 (which the human also cut) → auto-cut 0.67 → 0.75.
    assert by_val[3.0]["auto_cut_accuracy"] == pytest.approx(0.75)
    assert by_val[3.0]["n_gate_cut"] == 4
    assert by_val[3.5]["auto_cut_accuracy"] == pytest.approx(2 / 3)
    assert by_val[3.5]["baseline"] is True
    assert by_val[4.0]["auto_cut_accuracy"] == pytest.approx(2 / 3)


# --- per-axis contribution -------------------------------------------------------------------


def test_per_axis_contribution_counts():
    axes = _report()["per_axis"]
    assert axes["a"]["dominant_count"] == 6
    assert axes["b"]["dominant_count"] == 4
    # axis `a` drove 3 human-keeps, 2 human-cuts, 1 human-listen.
    assert axes["a"]["human"] == {"keep": 3, "listen": 1, "cut": 2}


# --- schema staleness ------------------------------------------------------------------------


def test_stale_schema_flagged(tmp_path):
    corpus = [
        {"id": "stale", "role": "kick", "features": {"a": 0.1, "b": 0.1},
         "human_verdict": "keep", "feature_schema": "smplstream/0-OLD"},
    ]
    profile, meta = V.load_profile(str(SEED_PROFILE))
    r = B.run_backtest(corpus, {"profile": profile, "meta": meta})
    assert r["stale_schema_ids"] == ["stale"]


# --- skipping (no profile / no human) --------------------------------------------------------


def test_entries_without_profile_or_human_are_skipped():
    corpus = [
        {"id": "no-human", "role": "kick", "features": {"a": 0.1}},
        {"id": "no-profile", "role": "snare", "features": {"a": 0.1}, "human_verdict": "keep"},
        {"id": "ok", "role": "kick", "features": {"a": 0.1, "b": 0.1}, "human_verdict": "keep"},
    ]
    profile, meta = V.load_profile(str(SEED_PROFILE))
    # A dict keyed by the one role we have a profile for; `snare` won't resolve.
    r = B.run_backtest(corpus, {"kick": (profile, meta)})
    assert r["n_scored"] == 1
    reasons = {s["id"]: s["reason"] for s in r["skipped"]}
    assert reasons["no-human"] == "no_human_verdict"
    assert reasons["no-profile"] == "no_profile"


# --- features_for_entry: precomputed AND sample_path -----------------------------------------


def test_features_for_entry_precomputed():
    feat = B.features_for_entry({"id": "x", "features": {"a": 0.5, "b": -1.0}})
    assert feat == {"a": 0.5, "b": -1.0}


def test_features_for_entry_from_sample_path(tmp_path, monkeypatch):
    # Prove the corpus is runnable from audio too: synthesize a wav, resolve features through the
    # same describe path the gate uses at judgment time.
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    sr = 44100
    y = (0.3 * np.sin(2 * np.pi * 110.0 * np.arange(sr) / sr)).astype("float32")
    wav = tmp_path / "tone.wav"
    with io.BytesIO() as buf:
        sf.write(buf, y.reshape(-1, 1), sr, format="WAV", subtype="FLOAT")
        wav.write_bytes(buf.getvalue())
    feat = B.features_for_entry({"id": "audio", "role": "kick", "sample_path": str(wav)})
    assert isinstance(feat, dict) and len(feat) > 0
    assert all(isinstance(v, float) for v in feat.values())


# --- report renders -------------------------------------------------------------------------


def test_format_report_is_legible():
    text = B.format_report(_report())
    assert "confusion matrix" in text
    assert "auto-keep accuracy" in text
    assert "BELOW TARGET" in text
    assert "threshold sensitivity" in text
