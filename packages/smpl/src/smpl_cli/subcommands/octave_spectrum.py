"""`smpl octave-spectrum` — the 1/6-octave spectrum feature as a filter (ticket vault-22oy).

Passes every input frame through unchanged, then for each selected `audio` frame appends a
`feature` frame (role ``octave-spectrum``) carrying the level-normalized 1/6-octave spectrum
(``spectrum.oct6.<hz>``) plus the six standardized band levels (``spectrum.band.<name>``).

Feed it into ``smpl stats build`` to reduce a role corpus to a median spectrum + p10/p90 band
that ``smpl profile-overlay`` then draws behind a candidate:

    smpl read pads/*.wav | smpl octave-spectrum | smpl stats build --role pad

Thin wrapper: the DSP lives in `smpl_analysis.octave`. Heavy imports stay inside `run()`.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "1/6-octave spectrum feature frame per audio frame (spectrum.oct6.* + spectrum.band.*)"


def add_arguments(parser):
    add_selection_args(parser)


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    try:
        from smpl_analysis import octave as _octave
    except Exception as exc:  # analysis tier not installed
        eprint(f"octave-spectrum: analysis library unavailable: {exc}")
        emit(out)
        return 1

    rc = 0
    for audio in audios:
        try:
            out.extend(_octave.octave_audio_frame(audio))
        except Exception as exc:
            eprint(f"octave-spectrum: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="octave-spectrum"))
            rc = 1
    emit(out)
    return rc
