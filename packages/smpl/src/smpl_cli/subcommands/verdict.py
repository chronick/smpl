"""`smpl verdict` — the gen-QC judge: emit a `verdict` frame per candidate (ticket vault-1apl).

Reads an analysis stream (candidate `audio` frames plus their describe/feature frames) and a
role stats file, then appends ONE `verdict` frame per candidate: discretized per-axis verdicts
(5-token enum) + a routing rollup (keep | listen | cut | alter) conforming to
``music/smplstream/verdict-frames.md``. The human ear is spent only on the low-confidence
middle band (listen).

Typical use:

    smpl read cand/*.wav | smpl describe-all --all --no-image \\
        | smpl verdict --stats ~/.smpl/stats/kick.json

The scoring + routing live in `smpl_analysis.verdict` (which reuses `smpl_analysis.triage`);
this module is selection + emit glue. Thin wrapper — heavy imports stay inside `run()`.
"""

from __future__ import annotations

from .._common import emit, eprint, read_stdin_frames

HELP = "gen-QC judge: emit a verdict frame per candidate (keep/listen/cut/alter) vs a role profile"


def add_arguments(parser):
    parser.add_argument("--stats", required=True,
                        help="path to a role stats JSON (from `smpl stats build`)")
    parser.add_argument("--role", default=None,
                        help="only treat audio frames with this role as candidates")
    parser.add_argument("--target", default=None,
                        help="optional absolute-tolerance brief JSON ({key:[lo,hi]}); "
                             "adds per-axis in_tolerance")


def run(args) -> int:
    from smplstream import error_frame

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    try:
        from smpl_analysis import verdict as _verdict
    except Exception as exc:  # analysis tier not installed
        eprint(f"verdict: analysis library unavailable: {exc}")
        emit(out)
        return 1

    try:
        profile, meta = _verdict.load_profile(args.stats)
    except Exception as exc:
        eprint(f"verdict: cannot read stats {args.stats!r}: {exc}")
        emit(out)
        return 1

    target = None
    if args.target:
        try:
            target = _verdict.load_target(args.target)
        except Exception as exc:
            eprint(f"verdict: cannot read target {args.target!r}: {exc}")
            emit(out)
            return 1

    rc = 0
    try:
        derived = _verdict.verdict_frames(inframes, profile, meta, role=args.role, target=target)
        if not derived:
            eprint("verdict: no candidate audio frames in the stream")
        out.extend(derived)
        for d in derived:
            r = (d.get("data") or {}).get("rollup") or {}
            label = str(d.get("role") or d.get("id"))
            eprint(f"verdict: {label}: {r.get('decision')} "
                   f"({r.get('confidence')}, dom={r.get('dominant_axis')}, "
                   f"dist={r.get('distance')})")
    except Exception as exc:
        eprint(f"verdict: {exc}")
        out.append(error_frame("op_failed", str(exc), op="verdict"))
        rc = 1

    emit(out)
    return rc
