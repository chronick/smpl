"""`smpl write <out>` — materialize the selected audio frame to a file (a tee-style sink).

Passes frames through to stdout by default so `… | smpl write out.wav | …` keeps composing;
`--quiet` suppresses the passthrough.
"""

from __future__ import annotations

from pathlib import Path

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "write the selected audio frame to a file"


def add_arguments(parser):
    parser.add_argument("out", help="output file path")
    add_selection_args(parser)
    parser.add_argument("--format", help="output subtype (e.g. PCM_16, PCM_24, FLOAT); default: faithful copy")
    parser.add_argument("--quiet", action="store_true", help="do not pass frames through to stdout")


def run(args) -> int:
    from smplstream import cas, select as S

    inframes = read_stdin_frames()
    audio = S.resolve_single_audio(inframes, role=args.role, strict=args.strict)
    src = cas.get_path(audio["hash"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    same_container = src.suffix.lower() == out.suffix.lower()
    if same_container and not args.format:
        import shutil

        shutil.copyfile(src, out)
    else:
        import soundfile as sf

        data, sr = sf.read(str(src), dtype="float32", always_2d=False)
        subtype = args.format.upper() if args.format else None
        sf.write(str(out), data, sr, subtype=subtype)
    eprint(f"write: {out} ({audio['meta'].get('dur', 0):.3f}s)")

    if not args.quiet:
        emit(inframes)
    return 0
