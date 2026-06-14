"""`smpl select` — filter the frame stream by role/kind (ticket vault-3l83).

A pure stream filter: resolve the matching frames via ``smplstream.select.select`` (last-wins
by default, ``--all`` for every match, ``--strict`` to error on >1). Per the tool contract,
every input frame is passed through unchanged (the tail of a pipe must still see the whole
lineage); ``select`` annotates the matches with a ``role: selected`` marker frame so a
downstream consumer can see *which* frames the predicate picked without reordering the
stream. There are no heavy imports here — selection is pure smplstream.
"""

from __future__ import annotations

from .._common import emit, eprint, read_stdin_frames, selection_mode

HELP = "filter the frame stream by --role / --kind (last-wins / --all / --strict)"


def add_arguments(parser):
    parser.add_argument("--role", help="match frames with this role")
    parser.add_argument("--kind", help="match frames of this kind (audio/feature/marker/...)")
    parser.add_argument("--all", dest="select_all", action="store_true",
                        help="match every frame, not just the last")
    parser.add_argument("--strict", action="store_true", help="error if more than one matches")
    parser.add_argument("--ids-only", dest="ids_only", action="store_true",
                        help="don't passthrough; emit only the matched frames")


def run(args) -> int:
    from smplstream import error_frame, select as S
    from smplstream.errors import ResolutionError

    inframes = read_stdin_frames()

    if args.role is None and args.kind is None:
        eprint("select: need at least one of --role / --kind")
        return 2

    try:
        matched = S.select(
            inframes, role=args.role, kind=args.kind, mode=selection_mode(args)
        )
    except ResolutionError as exc:
        eprint(f"select: {exc}")
        emit(list(inframes) + [error_frame("op_failed", str(exc), op="select")])
        return 1

    matched_ids = {m.get("id") for m in matched}

    if args.ids_only:
        # Consuming mode: drop everything that didn't match. Lineage to the matches is
        # carried by the matched frames themselves (they keep their own of/lineage).
        emit(matched)
        return 0

    # Default: passthrough everything (tool contract), then a marker frame naming the
    # matches so a downstream stage knows the selection without reordering the stream.
    from smplstream import frames as F

    out = list(inframes)
    points = [{"label": mid} for mid in matched_ids if mid]
    out.append(
        F.marker_frame(
            points,
            role="selected",
            op="select",
            op_version="select@1",
            params={"role": args.role, "kind": args.kind, "mode": selection_mode(args),
                    "matched": sorted(mid for mid in matched_ids if mid)},
        )
    )
    emit(out)
    return 0
