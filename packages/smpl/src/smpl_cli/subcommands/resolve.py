"""`smpl resolve <id|hash>` — print the CAS filesystem path for a blob (the "hand a path
to anything" seam: feed Demucs, a VST renderer, Audacity, sox).

Accepts a CAS hash directly, a frame id (resolved against the stdin frame stream), or
`--role R` (selected from the stdin stream, last-wins).
"""

from __future__ import annotations

from .._common import add_selection_args, eprint, read_stdin_frames

HELP = "print the CAS path for a hash, frame id, or selected role"


def add_arguments(parser):
    parser.add_argument("target", nargs="?", help="a CAS hash (blake3:…) or a frame id")
    add_selection_args(parser)


def run(args) -> int:
    from smplstream import cas, select as S

    # Direct hash → straight to a path (no stream needed).
    if args.target and cas.HASH_RE.match(args.target):
        print(cas.get_path(args.target))
        return 0

    inframes = read_stdin_frames()
    if not inframes:
        eprint("resolve: no frame stream on stdin; pass a CAS hash or pipe frames")
        return 1

    if args.target:  # resolve a frame id → its hash → path
        match = next((f for f in inframes if f.get("id") == args.target), None)
        if match is None or not match.get("hash"):
            eprint(f"resolve: no resolvable blob for id {args.target}")
            return 1
        print(cas.get_path(match["hash"]))
        return 0

    # No target → select an audio frame by role (last-wins) and print its path.
    audio = S.resolve_single_audio(inframes, role=args.role, strict=args.strict)
    print(cas.get_path(audio["hash"]))
    return 0
