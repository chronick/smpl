"""`smpl read <files...>` — CAS audio files and emit `audio` frames (the head of a pipe)."""

from __future__ import annotations

from .._common import emit, eprint, read_stdin_frames

HELP = "ingest audio file(s) into the CAS and emit audio frames"

OP_VERSION = "read@1"


def add_arguments(parser):
    parser.add_argument("files", nargs="+", help="audio file(s) to ingest")
    parser.add_argument("--role", default="source", help="role for emitted frames (default: source)")


def run(args) -> int:
    from smplstream import cas, error_frame, frames as F

    out = list(read_stdin_frames())  # passthrough any inbound stream first
    rc = 0
    for path in args.files:
        try:
            h = cas.put_audio_file(path)
            meta = cas.read_meta(h) or {}
            out.append(
                F.audio_frame(
                    h,
                    sr=meta.get("sr", 0),
                    ch=meta.get("ch", 1),
                    dur=meta.get("dur", 0.0),
                    role=args.role,
                    op="read",
                    op_version=OP_VERSION,
                    fmt=meta.get("fmt"),
                    params={"source": path},
                )
            )
        except Exception as exc:  # one file failing shouldn't abort the batch
            eprint(f"read: {path}: {exc}")
            out.append(error_frame("decode_failed", f"{path}: {exc}", op="read"))
            rc = 1
    emit(out)
    return rc
