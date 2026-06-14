"""`smpl env` — amplitude envelope (pluck / fade / gate) on the selected audio frame
(ticket vault-3l83).

Passthrough every input frame first, then append one wet `audio` frame (role ``<role>.wet``,
``op: env``) per selected audio frame. DSP lives in ``smpl_analysis.edit``.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "apply pluck/fade/gate amplitude envelope (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pluck", action="store_true", help="fast attack + exponential decay")
    group.add_argument("--fade", action="store_true", help="linear fade in/out")
    group.add_argument("--gate", action="store_true", help="silence below the threshold")
    parser.add_argument("--attack", type=float, default=0.005, help="attack seconds (default 0.005)")
    parser.add_argument("--release", type=float, default=0.2, help="release/decay seconds (default 0.2)")
    parser.add_argument("--threshold-db", dest="threshold_db", type=float, default=-40.0,
                        help="gate threshold in dBFS (default -40)")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)

    if args.pluck:
        shape = "pluck"
    elif args.fade:
        shape = "fade"
    else:
        shape = "gate"

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import edit

    rc = 0
    for audio in audios:
        try:
            out.append(edit.apply_env(
                audio, shape=shape, attack=args.attack, release=args.release,
                threshold_db=args.threshold_db,
            ))
        except Exception as exc:
            eprint(f"env: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="env"))
            rc = 1
    emit(out)
    return rc
