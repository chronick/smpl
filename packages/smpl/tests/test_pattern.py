"""Unit tests for `smpl pattern` DSL expansion (smpl_cli.subcommands.pattern._expand).

Covers the per-bar hit gating (`bar` / `bars`) that lifts the 1-bar-repeat
ceiling, plus the baseline repeat-across-bars behavior it must stay compatible with.
"""

from __future__ import annotations

import pytest

from smpl_cli.subcommands.pattern import _expand


def _clip_bars(session, track_name, bpb=4.0):
    """Return the set of bar numbers (1-indexed) a track's clips land on."""
    trk = next(t for t in session["tracks"] if t["name"] == track_name)
    bars = set()
    for c in trk["clips"]:
        bar = int(c["at"].split(".")[0])
        bars.add(bar)
    return bars, len(trk["clips"])


def test_repeats_across_bars_by_default():
    """A hit with no bar/bars fires on every bar (backward compatible)."""
    s = _expand({"bpm": 130, "bars": 2, "grid_steps": 16,
                 "tracks": [{"name": "kick", "source": "k.wav",
                             "steps": [1, 5, 9, 13], "velocity": 1.0}]})
    bars, n = _clip_bars(s, "kick")
    assert bars == {1, 2}
    assert n == 8  # 4 hits x 2 bars


def test_bar_gates_single_bar():
    """`bar: 2` fires only on bar 2."""
    s = _expand({"bpm": 130, "bars": 2, "grid_steps": 16,
                 "tracks": [{"name": "ghost", "source": "k.wav",
                             "hits": [{"step": 16, "velocity": 0.5, "bar": 2}]}]})
    bars, n = _clip_bars(s, "ghost")
    assert bars == {2}
    assert n == 1


def test_bars_list_gates_multiple():
    """`bars: [1, 3]` fires on bars 1 and 3 only (not 2 or 4)."""
    s = _expand({"bpm": 130, "bars": 4, "grid_steps": 16,
                 "tracks": [{"name": "stab", "source": "s.wav",
                             "hits": [{"step": 4, "velocity": 0.8, "bars": [1, 3]}]}]})
    bars, n = _clip_bars(s, "stab")
    assert bars == {1, 3}
    assert n == 2


def test_bar_and_default_hits_coexist_on_one_track():
    """A steady hit (every bar) + a bar-2-only fill on the same track."""
    s = _expand({"bpm": 130, "bars": 2, "grid_steps": 16,
                 "tracks": [{"name": "perc", "source": "p.wav",
                             "hits": [{"step": 1, "velocity": 0.8},          # both bars
                                      {"step": 15, "velocity": 0.6, "bar": 2}]}]})  # bar 2 only
    bars, n = _clip_bars(s, "perc")
    assert bars == {1, 2}
    assert n == 3  # step1 x2 bars + step15 x1


def test_bar_out_of_range_is_silent_not_error():
    """A bar number beyond `bars` simply never fires (no crash)."""
    s = _expand({"bpm": 130, "bars": 2, "grid_steps": 16,
                 "tracks": [{"name": "x", "source": "x.wav",
                             "hits": [{"step": 1, "bar": 5}]}]})
    trk = next(t for t in s["tracks"] if t["name"] == "x")
    assert trk["clips"] == []
