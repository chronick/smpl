"""`smpl limit` — true-peak ceiling on the selected audio frame (level-management trio).

Guarantees true peak ≤ ``--ceiling`` dBTP by whole-sample gain reduction (never boosts) — a
transparent, deterministic ceiling, not a look-ahead compressor. The safety stage for one-
shots and the tail of a normalize chain. Passthrough every input frame, then append one wet
`audio` frame (role ``<role>.wet``, ``op: limit``) per selected audio frame. DSP lives in
``smpl_analysis.edit``.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "true-peak ceiling via gain reduction (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--ceiling", type=float, default=-1.0,
                        help="true-peak ceiling in dBTP (default -1.0)")


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
            out.append(edit.apply_limit(audio, ceiling_dbtp=args.ceiling))
        except Exception as exc:
            eprint(f"limit: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="limit"))
            rc = 1
    emit(out)
    return rc
