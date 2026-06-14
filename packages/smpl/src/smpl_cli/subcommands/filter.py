"""`smpl filter` — high/low/band-pass the selected audio frame (ticket vault-3l83).

Passthrough every input frame first, then append one wet `audio` frame (role ``<role>.wet``,
``op: filter``) per selected audio frame. Filtering is scipy Butterworth (pure-Python,
deterministic). The DSP lives in ``smpl_analysis.edit``; this module is thin.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "high/low/band-pass filter the selected audio frame (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hp", type=float, metavar="HZ", help="high-pass cutoff (Hz)")
    group.add_argument("--lp", type=float, metavar="HZ", help="low-pass cutoff (Hz)")
    group.add_argument("--bp", type=float, nargs=2, metavar=("LOW", "HIGH"),
                       help="band-pass low/high cutoffs (Hz)")
    parser.add_argument("--order", type=int, default=4, help="filter order (default 4)")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    if args.hp is not None:
        kind, freq = "hp", args.hp
    elif args.lp is not None:
        kind, freq = "lp", args.lp
    else:
        kind, freq = "bp", args.bp

    from smpl_analysis import edit

    rc = 0
    for audio in audios:
        try:
            out.append(edit.apply_filter(audio, kind=kind, freq=freq, order=args.order))
        except Exception as exc:
            eprint(f"filter: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="filter"))
            rc = 1
    emit(out)
    return rc
