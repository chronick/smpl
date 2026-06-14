"""`smpl fx` — reverb / delay on the selected audio frame via sox (ticket vault-3l83).

Passthrough every input frame first, then append one wet `audio` frame (role ``<role>.wet``,
``op: fx``) per selected audio frame. sox is shelled out and its version fingerprinted into
``params`` (spec → *Memoization* / ``env_fingerprint``). DSP lives in ``smpl_analysis.edit``.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "apply reverb/delay via sox (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reverb", type=float, nargs="?", const=50.0, metavar="AMOUNT",
                       help="reverb amount 0-100 (default 50)")
    group.add_argument("--delay", action="store_true", help="single-tap echo/delay")
    parser.add_argument("--delay-ms", dest="delay_ms", type=float, default=250.0,
                        help="delay time in ms (default 250)")
    parser.add_argument("--decay", type=float, default=0.5, help="delay feedback decay (default 0.5)")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)

    if args.reverb is not None:
        effect, kwargs = "reverb", {"amount": args.reverb}
    else:
        effect, kwargs = "delay", {"delay_ms": args.delay_ms, "decay": args.decay}

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import edit

    rc = 0
    for audio in audios:
        try:
            out.append(edit.apply_fx(audio, effect=effect, **kwargs))
        except Exception as exc:
            eprint(f"fx: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="fx"))
            rc = 1
    emit(out)
    return rc
