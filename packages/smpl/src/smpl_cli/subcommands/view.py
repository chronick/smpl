"""`smpl view` — the terminal presentation (ticket vault-31r9).

Reads the accumulated frame stream, renders one consolidated LLM/human-facing
markdown report, and emits it as a single `text` frame (role ``report``) — after
passing **every** input frame through unchanged (so images/audio stay resolvable
downstream). The human-readable report is ALSO printed to stderr.

The report (markdown) tables every `feature` key/value with units, lists
`marker` frames (count + first few times), lists `image` frames with their role
AND resolved CAS path (``cas.get_path``) so an LLM can open the PNG, and
surfaces any `error` frames prominently at the top.

This module is a thin shim; the rendering lives in ``smpl_analysis.view``.
"""

from __future__ import annotations

from .._common import emit, eprint, read_stdin_frames

HELP = "render a consolidated markdown report over the frame stream (passthrough + text/report)"


def add_arguments(parser):
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the human-readable report to stderr (still emits the report frame)",
    )


def run(args) -> int:
    from smplstream import error_frame

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    try:
        from smpl_analysis import view as _view  # light import (no librosa/matplotlib)
    except Exception as exc:  # analysis tier not installed
        eprint(f"view: analysis library unavailable: {exc}")
        emit(out)
        return 1

    rc = 0
    try:
        derived = _view.view_frames(inframes)
        out.extend(derived)
        if not args.quiet:
            # Echo the rendered report to stderr for the human (stdout stays frames-only).
            for d in derived:
                body = d.get("data")
                if isinstance(body, str):
                    eprint(body)
                else:
                    # Report moved to CAS (oversized); point the human at the path.
                    from smplstream import cas

                    try:
                        eprint(f"view: report stored in CAS at {cas.get_path(d.get('hash'))}")
                    except Exception:
                        eprint(f"view: report stored in CAS as {d.get('hash')}")
    except Exception as exc:
        eprint(f"view: {exc}")
        out.append(error_frame("op_failed", str(exc), op="view"))
        rc = 1

    emit(out)
    return rc
