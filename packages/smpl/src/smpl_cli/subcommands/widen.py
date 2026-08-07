"""`smpl widen` — mid-side stereo widening above a crossover (reference-match family).

Scales the side channel ABOVE ``--crossover`` by ``--side-gain-db``, leaving the low end
(below the crossover) untouched so the bass stays mono-compatible. Passthrough every input
frame first, then append one wet `audio` frame (role ``<role>.wet``, ``op: widen``) per
selected audio frame. DSP lives in ``smpl_analysis.edit``; this module is thin.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "M-S widen the selected audio above a crossover, mono-safe low end (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--side-gain-db", type=float, default=3.0,
                        help="dB applied to the side channel above the crossover (negative narrows; default 3)")
    parser.add_argument("--crossover", type=float, default=200.0,
                        help="crossover Hz; the side below this is left untouched (default 200)")
    parser.add_argument("--order", type=int, default=4, help="crossover filter order (default 4)")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import edit

    rc = 0
    for audio in audios:
        try:
            out.append(edit.apply_widen(
                audio, side_gain_db=args.side_gain_db, crossover_hz=args.crossover, order=args.order
            ))
        except Exception as exc:
            eprint(f"widen: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="widen"))
            rc = 1
    emit(out)
    return rc
