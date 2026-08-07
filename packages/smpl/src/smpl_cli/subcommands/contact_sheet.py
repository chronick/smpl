"""`smpl contact-sheet` — batch-triage composite PNG (ticket vault-22oy).

Reads an analysis stream (candidate `audio` frames plus their describe/feature frames) and a
role stats file, then appends ONE composite `image` frame: mel tiles sorted best→worst by
aggregate robust-z distance from the profile, each with a burned-in verdict + worst-axes and a
border colored by verdict band. Also appends a `text` index (role ``contact-sheet-index``) —
the same ranking as markdown, so the deterministic order is legible without opening the PNG.

Typical use:

    smpl read cand/*.wav | smpl describe-all | smpl contact-sheet --stats ~/.smpl/stats/kick.json

The sorting/scoring live in `smpl_analysis.triage`; this module is selection + emit glue.
"""

from __future__ import annotations

from .._common import emit, eprint, read_stdin_frames

HELP = "composite triage contact sheet: mel tiles sorted best→worst by robust-z vs a role profile"


def add_arguments(parser):
    parser.add_argument("--stats", required=True,
                        help="path to a role stats JSON (from `smpl stats build`)")
    parser.add_argument("--role", default=None,
                        help="only treat audio frames with this role as candidates")
    parser.add_argument("--cols", type=int, default=None, help="tiles per row (default: min(4, n))")
    parser.add_argument("--title", default=None, help="override the sheet title")


def run(args) -> int:
    from smplstream import error_frame

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    try:
        from smpl_analysis import triage as _triage
    except Exception as exc:  # analysis tier not installed
        eprint(f"contact-sheet: analysis library unavailable: {exc}")
        emit(out)
        return 1

    try:
        profile = _triage.load_stats_file(args.stats)
    except Exception as exc:
        eprint(f"contact-sheet: cannot read stats {args.stats!r}: {exc}")
        emit(out)
        return 1

    rc = 0
    try:
        derived = _triage.render_contact_sheet(
            inframes, profile, role=args.role, cols=args.cols, title=args.title
        )
        if not derived:
            eprint("contact-sheet: no candidate audio frames in the stream")
        out.extend(derived)
        for d in derived:
            if d.get("role") == "contact-sheet-index" and isinstance(d.get("data"), str):
                eprint(d["data"])
    except Exception as exc:
        eprint(f"contact-sheet: {exc}")
        out.append(error_frame("op_failed", str(exc), op="contact-sheet"))
        rc = 1

    emit(out)
    return rc
