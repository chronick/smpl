"""`smpl stereoize` — decorrelation widener: synthesise stereo width from a mono(-ish) source.

Unlike `widen` (which only scales EXISTING side), stereoize injects an allpass-decorrelated copy
of the high-band mid into the side channel, so the mono downmix is preserved bit-for-bit
(mono-compatible; mono-measured spectral balance untouched) while side/mid rises. The right tool
for widening a synth/mono palette without changing its timbre. DSP lives in ``smpl_analysis.edit``.

  smpl read mix.wav | smpl stereoize --amount 0.6 --crossover 200 | smpl write wide.wav
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "widen a mono(-ish) source via decorrelation, mono-sum preserved (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--amount", type=float, default=0.4,
                        help="decorrelated side injection 0..~1+ (0 = no-op; default 0.4)")
    parser.add_argument("--crossover", type=float, default=200.0,
                        help="keep the low end below this mono (default 200)")
    parser.add_argument("--order", type=int, default=4, help="crossover filter order (default 4)")


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
            out.append(edit.apply_stereoize(
                audio, amount=args.amount, crossover_hz=args.crossover, order=args.order))
        except Exception as exc:
            eprint(f"stereoize: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="stereoize"))
            rc = 1
    emit(out)
    return rc
