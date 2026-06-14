"""Shared CLI helpers: stdin frame reading, passthrough, selection args, stderr logging."""

from __future__ import annotations

import sys
from typing import Iterable, Optional

from smplstream import ndjson


def eprint(*args, **kwargs) -> None:
    """Log to stderr (stdout is reserved for frames / raw bytes)."""
    print(*args, file=sys.stderr, **kwargs)


def stdin_is_pipe() -> bool:
    try:
        return not sys.stdin.isatty()
    except (ValueError, OSError):
        return True


def read_stdin_frames() -> list[dict]:
    """Read NDJSON frames from stdin, or [] when stdin is an interactive tty.

    Per the spec (*Id assignment*), if two inbound frames share an id we MUST emit an
    `id_collision` error frame; we do so eagerly on stdout so the failure is visible
    downstream rather than silently minting ambiguous references.
    """
    if not stdin_is_pipe():
        return []
    frames = list(ndjson.read_frames())
    from smplstream import error_frame
    from smplstream.frames import find_duplicate_ids

    for dup in find_duplicate_ids(frames):
        ndjson.write_frame(error_frame("id_collision", f"two distinct inbound frames share id {dup}"))
    return frames


def emit(frames: Iterable[dict]) -> None:
    ndjson.write_frames(frames)
    sys.stdout.buffer.flush()


def add_selection_args(parser, *, default_role: Optional[str] = None) -> None:
    parser.add_argument("--role", default=default_role,
                        help="select frames with this role (last-wins if multiple)")
    parser.add_argument("--all", dest="select_all", action="store_true",
                        help="act on every matching frame, not just the last")
    parser.add_argument("--strict", action="store_true",
                        help="error if more than one frame matches")


def selection_mode(args) -> str:
    if getattr(args, "select_all", False):
        return "all"
    if getattr(args, "strict", False):
        return "strict"
    return "last"
