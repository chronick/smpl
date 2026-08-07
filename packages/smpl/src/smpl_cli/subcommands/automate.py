"""`smpl automate` — time-varying parameter modulation (LFO / breakpoint envelopes).

The "movement" op. Modulate gain (tremolo / sidechain pump), pan (auto-pan), or a resonant
low-pass cutoff (filter sweep / acid) over time — with an LFO shape OR an arbitrary breakpoint
curve, tempo-synced via ``--period-beats`` + ``--bpm`` (or free with ``--rate-hz`` / ``--cycles``).
Static loops read as loops; automation is what makes a section evolve. DSP lives in
``smpl_analysis.edit``; this module is thin.

Examples:
  # sidechain pump: duck gain to 35% on each beat, recovering over the beat (4-on-floor feel)
  smpl read bass.wav | smpl automate --target gain --shape saw-up --period-beats 1 --bpm 118 --depth 0.65
  # acid filter sweep over 2 bars, resonant
  smpl read bass.wav | smpl automate --target cutoff --shape tri --period-beats 8 --bpm 118 --lo 180 --hi 4500 --resonance 6
  # complex breakpoint riser on noise over one 2-bar loop
  smpl read noise.wav | smpl automate --target cutoff --shape points --points "0:0.05,0.8:0.4,1:1.0" --cycles 1 --lo 300 --hi 12000 --resonance 3
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "modulate gain/pan/cutoff over time (LFO or breakpoints, tempo-syncable; emits <role>.wet)"


def _parse_points(s: str):
    pts = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        t, v = chunk.split(":")
        pts.append((float(t), float(v)))
    if not pts:
        raise ValueError("--points needs at least one t:v pair")
    return pts


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--target", required=True, choices=["gain", "pan", "cutoff"],
                        help="parameter to modulate")
    parser.add_argument("--shape", default="sine",
                        choices=["sine", "tri", "saw-up", "saw-down", "square", "points"],
                        help="LFO shape, or 'points' for a breakpoint curve (default sine)")
    parser.add_argument("--points", help="breakpoints 't:v,t:v,...' over one cycle (t,v in 0..1)")
    # rate: exactly one of these (period-beats needs --bpm)
    parser.add_argument("--period-beats", type=float,
                        help="LFO period in beats (needs --bpm); 1 = one cycle per beat")
    parser.add_argument("--bpm", type=float, help="tempo for --period-beats")
    parser.add_argument("--rate-hz", type=float, help="LFO rate in Hz (free-running)")
    parser.add_argument("--cycles", type=float, help="explicit number of cycles across the clip")
    parser.add_argument("--depth", type=float, default=1.0,
                        help="0..1 modulation amount (gain duck depth / pan width)")
    parser.add_argument("--phase", type=float, default=0.0, help="phase offset 0..1 (default 0)")
    parser.add_argument("--duty", type=float, default=0.5, help="square-wave duty 0..1 (default 0.5)")
    parser.add_argument("--lo", type=float, default=200.0, help="cutoff sweep low Hz (default 200)")
    parser.add_argument("--hi", type=float, default=8000.0, help="cutoff sweep high Hz (default 8000)")
    parser.add_argument("--resonance", type=float, default=0.707,
                        help="cutoff filter Q (0.707 flat; 3-10 = acid resonance)")
    parser.add_argument("--hop", type=int, default=128,
                        help="cutoff sweep block size in samples (smaller = smoother; default 128)")


def run(args) -> int:
    from smplstream import error_frame, select as S

    if args.shape == "points" and not args.points:
        eprint("automate: --shape points requires --points"); return 2
    try:
        points = _parse_points(args.points) if args.points else None
    except Exception as exc:
        eprint(f"automate: bad --points: {exc}"); return 2

    rate_n = sum(x is not None for x in (args.cycles, args.rate_hz, args.period_beats))
    if rate_n == 0:
        eprint("automate: specify one of --cycles, --rate-hz, --period-beats"); return 2
    if rate_n > 1:
        eprint("automate: --cycles / --rate-hz / --period-beats are mutually exclusive"); return 2
    if args.period_beats is not None and not args.bpm:
        eprint("automate: --period-beats needs --bpm"); return 2

    inframes = read_stdin_frames()
    out = list(inframes)
    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import edit

    rc = 0
    for audio in audios:
        try:
            if args.cycles is not None:
                cycles = float(args.cycles)
            else:
                dur = float(audio.get("dur") or (audio.get("meta") or {}).get("dur") or 0.0)
                if dur <= 0:
                    raise ValueError("frame has no duration; use --cycles")
                if args.rate_hz is not None:
                    cycles = float(args.rate_hz) * dur
                else:
                    cycles = dur / (float(args.period_beats) * 60.0 / float(args.bpm))
            out.append(edit.apply_automate(
                audio, target=args.target, shape=args.shape, cycles=cycles,
                depth=args.depth, phase=args.phase, duty=args.duty, points=points,
                lo_hz=args.lo, hi_hz=args.hi, resonance=args.resonance, hop=args.hop))
        except Exception as exc:
            eprint(f"automate: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="automate"))
            rc = 1
    emit(out)
    return rc
