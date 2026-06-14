"""`smpl normalize` — loudness-normalize the selected audio frame to a target LUFS.

BS.1770 integrated-loudness normalize WITH a true-peak ceiling (default −1 dBTP), so it is
safe standalone and still composes (``read | normalize --lufs -14 | write``). Passthrough
every input frame, then append one wet `audio` frame (role ``<role>.wet``, ``op: normalize``)
per selected audio frame; the measured-in loudness + applied gain land in ``params``. DSP
lives in ``smpl_analysis.edit``.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "loudness-normalize to a target LUFS with a true-peak ceiling (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--lufs", type=float, required=True,
                        help="target integrated loudness (LUFS), e.g. -14")
    parser.add_argument("--ceiling", type=float, default=-1.0,
                        help="true-peak ceiling in dBTP (default -1.0); use --no-ceiling to disable")
    parser.add_argument("--no-ceiling", dest="no_ceiling", action="store_true",
                        help="normalize to exact LUFS with no true-peak ceiling")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)

    ceiling = None if getattr(args, "no_ceiling", False) else args.ceiling

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import edit

    rc = 0
    for audio in audios:
        try:
            out.append(edit.apply_normalize(audio, target_lufs=args.lufs, ceiling_dbtp=ceiling))
        except Exception as exc:
            eprint(f"normalize: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="normalize"))
            rc = 1
    emit(out)
    return rc
