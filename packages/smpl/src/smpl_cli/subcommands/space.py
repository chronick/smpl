"""`smpl space` — broadband mono-collapse penalty as a filter (ticket vault-1fxy).

Passes every input frame through unchanged, then for each selected `audio` frame appends a
`feature` frame (role ``space``) carrying ``space.mono_collapse_penalty_db`` — how much
loudness is lost when the stereo image sums to mono (0 dB = mono-safe, ~3 dB = decorrelated,
capped for anti-phase). The broadband complement to the per-band width family (vault-3tuy).

Thin wrapper: the DSP lives in `smpl_analysis.space`. Heavy imports stay inside `run()`.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "mono-collapse penalty feature frame per audio frame (broadband complement to width)"


def add_arguments(parser):
    add_selection_args(parser)


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import space as _space

    rc = 0
    for audio in audios:
        try:
            out.extend(_space.space_audio_frame(audio))
        except Exception as exc:
            eprint(f"space: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="space"))
            rc = 1
    emit(out)
    return rc
