"""`smpl width` — per-band stereo width / correlation as a filter (ticket vault-3tuy).

Passes every input frame through unchanged, then for each selected `audio` frame appends a
`feature` frame (role ``width``) carrying per-band Pearson L/R correlation and side/mid RMS
ratio for the six standardized bands (Sub/Bass/LoMid/Mid/UpperMid/Air) plus a full-band
reference — so a wide-sub mono-collapse a broadband number would hide is visible.

Thin wrapper: the DSP lives in `smpl_analysis.width`. Heavy imports stay inside `run()`.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "per-band stereo width/correlation feature frame per audio frame (sub..air + full)"


def add_arguments(parser):
    add_selection_args(parser)


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import width as _width

    rc = 0
    for audio in audios:
        try:
            out.extend(_width.width_audio_frame(audio))
        except Exception as exc:
            eprint(f"width: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="width"))
            rc = 1
    emit(out)
    return rc
