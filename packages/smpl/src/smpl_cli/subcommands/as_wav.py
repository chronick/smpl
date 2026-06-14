"""`smpl as-wav` — resolve the selected audio frame to a raw WAV byte stream on stdout.

The bridge OUT to the Unix DSP world: `… | smpl as-wav | sox - … | smpl from-wav …`.
Default output is float32 WAV at the frame's native rate + channel count (no silent
bit-depth/rate truncation); `--format` overrides. stdout is RAW BYTES here, not frames.
"""

from __future__ import annotations

import sys

from .._common import add_selection_args, read_stdin_frames

HELP = "resolve the selected audio frame to raw WAV bytes on stdout (sox/ffmpeg bridge)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--format", default="FLOAT",
                        help="WAV subtype: FLOAT (default), PCM_16, PCM_24, PCM_32")


def run(args) -> int:
    import io

    import soundfile as sf

    from smplstream import cas, select as S

    inframes = read_stdin_frames()
    audio = S.resolve_single_audio(inframes, role=args.role, strict=args.strict)
    src = cas.get_path(audio["hash"])

    data, sr = sf.read(str(src), dtype="float32", always_2d=False)
    # WAV back-patches its RIFF size header, so it needs a SEEKABLE sink — stdout is a
    # (non-seekable) pipe. Render to an in-memory buffer first, then emit complete bytes.
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype=args.format.upper())
    sys.stdout.buffer.write(buf.getvalue())
    sys.stdout.buffer.flush()
    return 0
