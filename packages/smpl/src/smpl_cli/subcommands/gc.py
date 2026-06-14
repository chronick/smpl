"""`smpl gc` — conservatively collect unreferenced CAS blobs.

v1 ships the SAFETY rule (never delete a live or in-flight blob): blobs referenced by
frames on stdin are kept, blobs within the grace window are reserved, an exclusive lock
excludes concurrent producers. Dry-run by default — pass `--yes` to actually delete.
"""

from __future__ import annotations

import sys

from .._common import eprint, read_stdin_frames

HELP = "garbage-collect unreferenced CAS blobs (dry-run unless --yes)"


def add_arguments(parser):
    parser.add_argument("--yes", action="store_true", help="actually delete (default: dry-run)")
    parser.add_argument("--grace", type=float, default=3600.0,
                        help="seconds; never delete blobs newer than this (default: 3600)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")


def _referenced_hashes(frames) -> set[str]:
    keep = set()
    for f in frames:
        if f.get("hash"):
            keep.add(f["hash"])
    return keep


def run(args) -> int:
    from smplstream import cas

    keep = _referenced_hashes(read_stdin_frames())
    summary = cas.gc(keep=keep, grace_seconds=args.grace, dry_run=not args.yes)

    if args.json:
        import json

        print(json.dumps(summary, indent=2))
    else:
        verb = "would remove" if summary["dry_run"] else "removed"
        eprint(
            f"gc: {verb} {len(summary['removed'])} blob(s); "
            f"kept {len(summary['kept'])} referenced, "
            f"reserved {len(summary['reserved_in_grace'])} in grace window"
        )
        if summary["dry_run"] and summary["removed"]:
            eprint("    (dry-run — pass --yes to delete)")
    sys.stdout.flush()
    return 0
