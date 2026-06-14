"""`smpl eq` — peaking / shelving EQ on the selected audio frame (ticket vault-3l83).

Each ``--peak``/``--lowshelf``/``--highshelf`` adds one band to the chain. Passthrough every
input frame first, then append one wet `audio` frame (role ``<role>.wet``, ``op: eq``) per
selected audio frame. DSP lives in ``smpl_analysis.edit``.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "peaking/shelf EQ the selected audio frame (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--peak", action="append", nargs=3, default=[],
                        metavar=("FREQ", "GAIN_DB", "Q"),
                        help="peaking band: freq(Hz) gain(dB) Q (repeatable)")
    parser.add_argument("--lowshelf", action="append", nargs=2, default=[],
                        metavar=("FREQ", "GAIN_DB"), help="low-shelf band: freq(Hz) gain(dB)")
    parser.add_argument("--highshelf", action="append", nargs=2, default=[],
                        metavar=("FREQ", "GAIN_DB"), help="high-shelf band: freq(Hz) gain(dB)")


def _build_bands(args) -> list[dict]:
    bands = []
    for freq, gain, q in args.peak:
        bands.append({"type": "peaking", "freq": float(freq), "gain": float(gain), "q": float(q)})
    for freq, gain in args.lowshelf:
        bands.append({"type": "lowshelf", "freq": float(freq), "gain": float(gain)})
    for freq, gain in args.highshelf:
        bands.append({"type": "highshelf", "freq": float(freq), "gain": float(gain)})
    return bands


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)

    bands = _build_bands(args)
    if not bands:
        eprint("eq: no bands given (use --peak/--lowshelf/--highshelf)")
        return 2

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import edit

    rc = 0
    for audio in audios:
        try:
            out.append(edit.apply_eq(audio, bands=bands))
        except Exception as exc:
            eprint(f"eq: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="eq"))
            rc = 1
    emit(out)
    return rc
