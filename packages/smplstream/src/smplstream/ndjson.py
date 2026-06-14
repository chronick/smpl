"""NDJSON stream I/O (spec → *The frame*, *Pipe hygiene*).

One frame per line, UTF-8, decoded to a **dict** (not a typed struct) so unknown fields
survive passthrough verbatim (forward-compat is normative). Lines up to >=1 MiB are
accepted. A truncated final line is a fatal read error. Tools must handle SIGPIPE cleanly
and never emit a truncated final NDJSON line.
"""

from __future__ import annotations

import sys
from typing import BinaryIO, Iterable, Iterator

import msgspec

from .errors import ProtocolError

_decoder = msgspec.json.Decoder()
_encoder = msgspec.json.Encoder()

# Accept generous line lengths (spec: >=1 MiB). Python's buffered readline has no hard
# cap, but we guard against a single absurdly long line eating memory unbounded.
_MAX_LINE_BYTES = 64 * 1024 * 1024


def read_frames(fileobj: BinaryIO | None = None) -> Iterator[dict]:
    """Yield frames (dicts) from an NDJSON byte stream (default: stdin)."""
    stream = fileobj if fileobj is not None else sys.stdin.buffer
    for raw in stream:
        if len(raw) > _MAX_LINE_BYTES:
            raise ProtocolError(f"NDJSON line exceeds {_MAX_LINE_BYTES} bytes")
        line = raw.strip()
        if not line:
            continue
        try:
            frame = _decoder.decode(line)
        except msgspec.DecodeError as exc:
            raise ProtocolError(f"invalid NDJSON frame: {exc}") from exc
        if not isinstance(frame, dict):
            raise ProtocolError("each NDJSON line must be a JSON object (frame)")
        yield frame


def write_frame(frame: dict, fileobj: BinaryIO | None = None) -> None:
    """Encode one frame as a single NDJSON line (msgspec preserves unknown keys)."""
    stream = fileobj if fileobj is not None else sys.stdout.buffer
    stream.write(_encoder.encode(frame))
    stream.write(b"\n")


def write_frames(frames: Iterable[dict], fileobj: BinaryIO | None = None) -> None:
    stream = fileobj if fileobj is not None else sys.stdout.buffer
    for frame in frames:
        write_frame(frame, stream)


def dumps(frame: dict) -> bytes:
    """Serialize a single frame to NDJSON bytes (no trailing newline)."""
    return _encoder.encode(frame)


def loads(line: bytes | str) -> dict:
    if isinstance(line, str):
        line = line.encode("utf-8")
    frame = _decoder.decode(line.strip())
    if not isinstance(frame, dict):
        raise ProtocolError("frame must be a JSON object")
    return frame
