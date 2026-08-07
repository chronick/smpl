"""`smpl maximize` — look-ahead brickwall limiter / loudness maximizer.

Drives the signal in by ``--makeup`` dB then peak-limits to ``--ceiling`` with a look-ahead window
(gain ducks BEFORE each transient, releases slowly) — closing the crest gap to a mastered reference
without the pumping of a compressor. Unlike `limit` (transparent whole-sample ceiling, preserves
crest) this REDUCES crest. DSP lives in ``smpl_analysis.edit``.

  smpl read mix.wav | smpl maximize --ceiling -1 --makeup 6 | smpl write master.wav
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "look-ahead brickwall limiter: drive + cap peaks, reduce crest (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--ceiling", type=float, default=-1.0, help="true-peak ceiling dBTP (default -1)")
    parser.add_argument("--makeup", type=float, default=6.0, help="input drive dB (default 6)")
    parser.add_argument("--lookahead", type=float, default=2.0, help="look-ahead ms (default 2)")
    parser.add_argument("--release", type=float, default=60.0, help="release ms (default 60)")


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
            out.append(edit.apply_maximize(
                audio, ceiling_dbtp=args.ceiling, makeup_db=args.makeup,
                lookahead_ms=args.lookahead, release_ms=args.release))
        except Exception as exc:
            eprint(f"maximize: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="maximize"))
            rc = 1
    emit(out)
    return rc
