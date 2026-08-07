"""`smpl stretch` — time-stretch at constant pitch (phase vocoder). Sampler edit primitive."""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "time-stretch by --ratio at constant pitch (>1 slower; emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--ratio", type=float, required=True, help="stretch factor (>1 slower)")


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
            out.append(edit.apply_stretch(audio, ratio=args.ratio))
        except Exception as exc:
            eprint(f"stretch: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="stretch"))
            rc = 1
    emit(out)
    return rc
