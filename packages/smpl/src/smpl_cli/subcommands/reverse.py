"""`smpl reverse` — reverse the selected audio frame in time (sampler edit primitive)."""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "reverse the selected audio in time (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)


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
            out.append(edit.apply_reverse(audio))
        except Exception as exc:
            eprint(f"reverse: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="reverse"))
            rc = 1
    emit(out)
    return rc
