"""`smpl pitch` — pitch-shift at constant duration (phase vocoder). For VOICE prefer
`smpl larynx render --semitones` (WORLD, formant-preserving); this is the timbre-agnostic edit."""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "pitch-shift by --semitones at constant duration (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--semitones", type=float, required=True, help="shift in semitones (±)")


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
            out.append(edit.apply_pitch(audio, semitones=args.semitones))
        except Exception as exc:
            eprint(f"pitch: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="pitch"))
            rc = 1
    emit(out)
    return rc
