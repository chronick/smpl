"""`smpl slice` — librosa onset detection → marker frame (+ optional sliced audio)
(ticket vault-3l83).

Passthrough every input frame first, then append a `marker` frame (role ``onset``, with
``t`` + sample-accurate ``sample`` per point) for each selected audio frame; with
``--emit-audio`` each inter-onset region is also CASed and emitted as a ``slice:<n>``
`audio` frame. DSP lives in ``smpl_analysis.edit``.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "onset-detect the selected audio frame into markers (+ optional sliced audio)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--emit-audio", dest="emit_audio", action="store_true",
                        help="also emit one sliced audio frame per region (role slice:<n>)")
    parser.add_argument("--no-backtrack", dest="backtrack", action="store_false",
                        help="don't backtrack onsets to the preceding energy minimum")


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
            out.extend(edit.slice_onsets(
                audio, emit_audio=args.emit_audio, backtrack=args.backtrack
            ))
        except Exception as exc:
            eprint(f"slice: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="slice"))
            rc = 1
    emit(out)
    return rc
