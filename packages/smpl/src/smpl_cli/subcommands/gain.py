"""`smpl gain` — apply a dB gain to the selected audio frame (level-management trio).

The level primitive: a pure dB scale with NO clipping (float-safe, so it composes — pair
with ``smpl limit`` for a delivery ceiling). Passthrough every input frame, then append one
wet `audio` frame (role ``<role>.wet``, ``op: gain``) per selected audio frame. DSP lives in
``smpl_analysis.edit``.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "apply a dB gain to the selected audio frame (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--db", type=float, required=True,
                        help="gain in dB (+ louder / - quieter); not clipped")


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
            out.append(edit.apply_gain(audio, db=args.db))
        except Exception as exc:
            eprint(f"gain: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="gain"))
            rc = 1
    emit(out)
    return rc
