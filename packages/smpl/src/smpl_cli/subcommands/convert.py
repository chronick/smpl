"""`smpl convert` — format / sample-rate / bit-depth conversion via ffmpeg (an explicit op).

Resolves the selected input audio frame, shells out to ffmpeg to re-encode at the requested
container `--format`, `--sr`, and `--bits`, CASes the result under a fresh hash, and appends
a new `audio` frame (`op: convert`, lineage → the input). Every input frame passes through
first (spec → *Tool contract*); the converted frame is appended after.

NB: this does not change how the source is hashed. The CAS audio hash is always canonical
decoded PCM (native-rate/native-channel/float32); converting to a storage norm makes a
*different* frame+hash — see the library module for the full note. The thin CLI just wires
selection + args into `smpl_analysis.convert`.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "convert audio format/sample-rate/bit-depth via ffmpeg (new frame + hash, op: convert)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--sr", type=int, default=None,
                        help="target sample rate in Hz (e.g. 44100, 48000); default keeps source rate")
    parser.add_argument("--bits", type=int, default=None, choices=[16, 24, 32],
                        help="target PCM bit depth; default keeps the container's natural depth")
    parser.add_argument("--format", dest="format", default=None,
                        help="target container format (wav|flac|aiff|mp3); default keeps source format")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")
    if not audios:
        eprint("convert: no audio frame on stdin to convert")
        emit(out)
        return 1

    try:
        from smpl_analysis import convert as _convert
    except Exception as exc:  # analysis tier not installed
        eprint(f"convert: analysis library unavailable: {exc}")
        emit(out)
        return 1

    rc = 0
    for audio in audios:
        try:
            derived = _convert.convert_audio_frame(
                audio, sr=args.sr, bits=args.bits, fmt=args.format
            )
        except Exception as exc:
            eprint(f"convert: {audio.get('id')}: {exc}")
            derived = [error_frame("op_failed", str(exc), of=audio.get("id"), op="convert")]
        out.extend(derived)
        if any(f.get("kind") == "error" for f in derived):
            rc = 1
    emit(out)
    return rc
