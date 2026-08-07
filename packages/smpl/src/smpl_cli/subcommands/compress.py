"""`smpl compress` — feed-forward downward compressor (crest reduction for loudness parity).

The level primitive gain/normalize/limit lacked: reduces the peak-to-RMS gap so a downstream
``normalize`` can push loudness UP toward a dense reference without breaching the true-peak
ceiling. Passthrough every input frame first, then append one wet `audio` frame (role
``<role>.wet``, ``op: compress``) per selected audio frame. DSP in ``smpl_analysis.edit``.

  smpl read loop.wav | smpl compress --threshold -18 --ratio 3 | smpl normalize --lufs -11 | smpl write out.wav
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "downward-compress the selected audio (crest reduction; emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--threshold", type=float, default=-18.0, help="threshold dBFS (default -18)")
    parser.add_argument("--ratio", type=float, default=3.0, help="compression ratio >=1 (default 3)")
    parser.add_argument("--attack", type=float, default=10.0, help="attack ms (default 10)")
    parser.add_argument("--release", type=float, default=120.0, help="release ms (default 120)")
    parser.add_argument("--knee", type=float, default=6.0, help="soft-knee width dB (default 6)")
    parser.add_argument("--makeup", type=float, default=0.0, help="make-up gain dB (default 0)")


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
            out.append(edit.apply_compress(
                audio, threshold_db=args.threshold, ratio=args.ratio, attack_ms=args.attack,
                release_ms=args.release, knee_db=args.knee, makeup_db=args.makeup))
        except Exception as exc:
            eprint(f"compress: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="compress"))
            rc = 1
    emit(out)
    return rc
