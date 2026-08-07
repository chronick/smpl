"""`smpl crop` — keep only a time window of the selected audio frame (sampler edit primitive)."""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "crop the selected audio to [--start, --end) seconds (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--start", type=float, default=0.0, help="start time (s)")
    parser.add_argument("--end", type=float, default=None, help="end time (s); omit = to the tail")


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
            out.append(edit.apply_crop(audio, start_s=args.start, end_s=args.end))
        except Exception as exc:
            eprint(f"crop: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="crop"))
            rc = 1
    emit(out)
    return rc
