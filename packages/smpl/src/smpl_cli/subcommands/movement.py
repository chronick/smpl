"""`smpl movement` — the movement feature family as a filter (ticket vault-1fxy).

Passes every input frame through unchanged, then for each selected `audio` frame appends a
`feature` frame (role ``movement``) carrying pump depth, per-band modulation depth, HF
activity, and end-of-loop tail decay. Duration-gated (null on material too short to support
the time-varying read), like LUFS-integrated.

Thin wrapper: the DSP lives in `smpl_analysis.movement`. Heavy imports stay inside `run()`.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "movement feature frame per audio frame (pump depth, band modulation, tail decay)"


def add_arguments(parser):
    add_selection_args(parser)


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import movement as _movement

    rc = 0
    for audio in audios:
        try:
            out.extend(_movement.movement_audio_frame(audio))
        except Exception as exc:
            eprint(f"movement: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="movement"))
            rc = 1
    emit(out)
    return rc
