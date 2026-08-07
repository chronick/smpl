"""`smpl pattern` — expand a compact step-grid drum-loop DSL into a smplmix session.

The LLM-/agent-friendly way to write an explicit drum loop: a 16-step (or N-step)
grid, per-track step lists, per-hit velocity + swing, instead of hand-placing every
clip on a ``bar.beat.frac`` timeline. Emits one ``feature`` frame whose ``data`` IS a
ready-to-render smplmix session (header + tracks[] of clips[]). Consume it with
``--out x.smplset.json`` (or ``| jq .data > x.smplset.json``) and render via
``smplmix render``. Composition still runs on smplmix (no Rust change); this is the
authoring front-end (parity epic vault-pehi).

DSL (JSON on ``--pattern-file`` or stdin)::

    {
      "bpm": 134,
      "beats_per_bar": 4,          # default 4
      "grid_steps": 16,            # steps per bar (default 16)
      "bars": 1,                   # pattern repeats across this many bars (default 1)
      "swing": 0.0,                # 0..1 global groove (see Swing)
      "key": "A", "scale": "minor",          # optional, passed to smplmix
      "master": {"loudnorm": true, "limiter": true},   # optional
      "tracks": [
        {"name": "kick", "source": "kick.wav", "steps": [1,5,9,13], "velocity": 1.0, "pitch": -2},
        {"name": "hat",  "source": "hat.wav",  "steps": [2,4,6,8,10,12,14,16],
                         "velocity": 0.7, "swing": 0.12},
        {"name": "tom",  "source": "tom.wav",  "len": "1/16",
                         "hits": [{"step": 9,  "velocity": 0.8, "pitch": 0},
                                  {"step": 11, "velocity": 0.9, "pitch": 3},
                                  {"step": 13, "velocity": 1.0, "pitch": 7},
                                  {"step": 15, "gain_db": -1.5, "pitch": 12, "nudge": 0.02}]}
      ]
    }

Each track names one sound (``source`` path, or ``hash``/``gen``) and places it on the
grid via ``steps`` (uniform velocity/pitch/swing) or ``hits`` (per-hit ``velocity``/
``gain_db``/``pitch``/``swing``/``len``/``nudge``). Steps are 1-indexed (1..grid_steps).

- **velocity** (0..1) → clip ``gain_db`` = 20·log10(velocity), clamped to [-40, 0]
  (1.0 = 0 dB; 0.5 ≈ -6 dB; 0.25 ≈ -12 dB). velocity ≤ 0 skips the hit (a rest).
- **gain_db** (per-hit, dB) sets the hit amplitude *explicitly* and overrides
  ``velocity`` for that hit (clamped [-60, +24]) — use when you want absolute levels
  rather than a 0..1 feel. (Track-level ``gain_db`` is the track fader; it stacks.)
- **pitch** (semitones, ± float) → clip ``transform.transpose_semitones`` — per-track
  default + per-hit override. Essential for tuned kicks, tom rolls, hat-pitch variation.
- **swing** (0..1) delays every *even* grid step (2,4,6,…) — the standard 16-step
  sequencer shuffle — by ``swing × stepGap`` beats (stepGap = beats_per_bar/grid_steps).
  Per-track/per-hit ``swing`` overrides the global value. Recommended ≤ 0.6.
- **nudge** (per-hit, beats) is an explicit ± timing offset added to any hit, for
  micro-timing the sequencer swing can't express.
- **bar** / **bars** (per-hit) restrict a hit to specific bar(s) of a multi-bar
  ``bars`` pattern (1-indexed): ``"bar": 2`` fires only on bar 2; ``"bars": [1,3]``
  on bars 1 and 3. Absent ⇒ every bar (backward compatible). This is how you break
  the 1-bar-repeat deadness — a ghost on bar 2, a fill on the last bar, an
  open/closed tone swap per bar — without leaving the step grid.

velocity/amplitude and pitch are the two levers that make a step pattern read as a
*played* drum loop rather than a metronome — both are first-class here.
"""

from __future__ import annotations

import json
import math
import sys

from .._common import emit, eprint, stdin_is_pipe

HELP = "expand a step-grid drum-loop DSL into a smplmix session (emits a feature frame)"

OP_VERSION = "pattern@1"


def add_arguments(parser):
    parser.add_argument("--pattern-file", help="pattern DSL JSON file (default: read stdin)")
    parser.add_argument("--out", help="also write the expanded smplmix session here (.smplset.json)")
    parser.add_argument("--role", default="smplset", help="role for the emitted feature frame (default: smplset)")


def _load_pattern(args) -> dict:
    if args.pattern_file:
        with open(args.pattern_file, encoding="utf-8") as fh:
            text = fh.read()
    elif stdin_is_pipe():
        text = sys.stdin.read()
    else:
        raise ValueError("no pattern: pass --pattern-file or pipe the DSL JSON on stdin")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("pattern must be a JSON object")
    return data


def _vel_to_db(vel) -> float | None:
    """velocity 0..1 → dB (1.0=0dB), clamped [-40,0]. None ⇒ skip (silent)."""
    if vel is None:
        vel = 1.0
    vel = min(float(vel), 1.0)
    if vel <= 0:
        return None
    db = 20.0 * math.log10(vel)
    return round(max(db, -40.0), 2)


def _beats_to_at(total_beats: float, bpb: float) -> str:
    """Total beats from the start → smplmix ``bar.beat.frac`` (1-indexed; frac = the
    decimal digits of the fractional beat, e.g. 0.25 → "25")."""
    if total_beats < 0:
        total_beats = 0.0
    bar = int(total_beats // bpb) + 1
    within = total_beats - (bar - 1) * bpb
    beat = int(within + 1e-9) + 1
    frac = within - int(within + 1e-9)
    if frac < 1e-6:
        return f"{bar}.{beat}"
    frac_str = f"{frac:.6f}".split(".")[1].rstrip("0")
    return f"{bar}.{beat}.{frac_str}"


def _build_source(track: dict) -> dict:
    """A smplmix Source object from a track's source spec (path | hash | gen)."""
    if track.get("gen"):
        return {"gen": track["gen"]}
    if track.get("hash"):
        return {"hash": track["hash"]}
    if track.get("path"):
        return {"path": track["path"]}
    src = track.get("source")
    if isinstance(src, dict):
        return src
    if isinstance(src, str):
        return {"hash": src} if src.startswith("blake3:") else {"path": src}
    raise ValueError(f"track '{track.get('name', '?')}' needs a source (source/path/hash/gen)")


def _normalize_hits(track: dict) -> list[dict]:
    if isinstance(track.get("hits"), list):
        return [h if isinstance(h, dict) else {"step": h} for h in track["hits"]]
    if isinstance(track.get("steps"), list):
        return [{"step": s} for s in track["steps"]]
    raise ValueError(f"track '{track.get('name', '?')}' needs `steps` or `hits`")


def _expand(p: dict) -> dict:
    bpm = float(p.get("bpm", 0) or 0)
    if bpm <= 0:
        raise ValueError("pattern needs a positive `bpm`")
    bpb = float(p.get("beats_per_bar", 4) or 4)
    grid = int(p.get("grid_steps", 16) or 16)
    bars = int(p.get("bars", 1) or 1)
    if bpb <= 0 or grid <= 0 or bars < 1:
        raise ValueError("beats_per_bar, grid_steps must be > 0 and bars >= 1")
    step_gap = bpb / grid
    g_swing = float(p.get("swing", 0.0) or 0.0)

    tracks_in = p.get("tracks") or []
    if not tracks_in:
        raise ValueError("pattern needs at least one track")

    seen_names: set[str] = set()
    tracks_out = []
    for t in tracks_in:
        name = str(t.get("name") or "")
        if not name:
            raise ValueError("every track needs a `name`")
        if name in seen_names:
            raise ValueError(f"duplicate track name '{name}' (smplmix requires unique names)")
        seen_names.add(name)
        src = _build_source(t)
        t_swing = float(t.get("swing", g_swing) or 0.0)
        t_vel = t.get("velocity", 1.0)
        t_len = t.get("len")
        t_pitch = float(t.get("pitch", 0.0) or 0.0)
        hits = _normalize_hits(t)

        clips = []
        for b in range(1, bars + 1):
            for h in hits:
                # Per-bar gating (movement / kill the 1-bar-repeat): a hit may name
                # the bar(s) it fires on via `bar` (int) or `bars` (list of ints,
                # 1-indexed). Absent ⇒ every bar (backward compatible). Lets you add
                # a ghost on bar 2, a fill on the last bar, an open/closed swap, etc.
                hbar = h.get("bar")
                if hbar is not None and b != int(hbar):
                    continue
                hbars = h.get("bars")
                if hbars is not None and b not in {int(x) for x in hbars}:
                    continue
                step = int(h["step"])
                if step < 1 or step > grid:
                    raise ValueError(f"track '{name}': step {step} out of range 1..{grid}")
                # Amplitude: explicit per-hit gain_db wins; else velocity (0..1).
                if h.get("gain_db") is not None:
                    db = round(max(min(float(h["gain_db"]), 24.0), -60.0), 2)
                else:
                    db = _vel_to_db(h.get("velocity", t_vel))
                    if db is None:
                        continue  # velocity <= 0 ⇒ rest
                sw = max(0.0, min(float(h.get("swing", t_swing) or 0.0), 0.95))
                within = (step - 1) * step_gap
                if step % 2 == 0:
                    within += sw * step_gap
                within += float(h.get("nudge", 0.0) or 0.0)
                clip: dict = {"source": src, "at": _beats_to_at((b - 1) * bpb + within, bpb)}
                ln = h.get("len", t_len)
                if ln is not None:
                    clip["len"] = ln
                if abs(db) > 1e-9:
                    clip["gain_db"] = db
                pitch = float(h.get("pitch", t_pitch) or 0.0)
                if abs(pitch) > 1e-9:
                    clip["transform"] = {"transpose_semitones": pitch}
                clips.append(clip)

        track_obj: dict = {"name": name, "clips": clips}
        if abs(float(t.get("gain_db", 0) or 0)) > 1e-9:
            track_obj["gain_db"] = float(t["gain_db"])
        if t.get("pan") is not None:
            track_obj["pan"] = float(t["pan"])
        tracks_out.append(track_obj)

    session: dict = {"bpm": bpm, "beats_per_bar": bpb, "bars": bars}
    for k in ("key", "key_ref", "scale", "sample_rate", "channels"):
        if p.get(k) is not None:
            session[k] = p[k]
    session["master"] = p.get("master", {"loudnorm": True, "limiter": True})
    session["tracks"] = tracks_out
    return session


def run(args) -> int:
    from smplstream import frames as F

    try:
        pattern = _load_pattern(args)
        session = _expand(pattern)
    except Exception as exc:
        eprint(f"pattern: {exc}")
        return 1

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(session, fh, indent=2)
                fh.write("\n")
            eprint(f"pattern: wrote {args.out}")
        except Exception as exc:
            eprint(f"pattern: could not write {args.out}: {exc}")
            return 1

    n_clips = sum(len(t["clips"]) for t in session["tracks"])
    emit([F.feature_frame(
        session,
        role=args.role,
        op="pattern",
        op_version=OP_VERSION,
        params={"tracks": len(session["tracks"]), "clips": n_clips,
                "grid_steps": int(pattern.get("grid_steps", 16) or 16),
                "bars": session["bars"]},
    )])
    return 0
