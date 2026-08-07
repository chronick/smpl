"""`smpl clarity` — the clarity feature family as a filter (ticket vault-1fxy).

Passes every input frame through unchanged, then for each selected `audio` frame appends a
`feature` frame (role ``clarity``) carrying the mud/presence balance, low-mid masking, low/high
band contrast, and presence focus/transient reads. Duration-gated like the movement family.

Thin wrapper: the DSP lives in `smpl_analysis.clarity`. Heavy imports stay inside `run()`.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "clarity feature frame per audio frame (mud/presence, low-mid masking, band contrast)"


def add_arguments(parser):
    add_selection_args(parser)


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import clarity as _clarity

    rc = 0
    for audio in audios:
        try:
            out.extend(_clarity.clarity_audio_frame(audio))
        except Exception as exc:
            eprint(f"clarity: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="clarity"))
            rc = 1
    emit(out)
    return rc
