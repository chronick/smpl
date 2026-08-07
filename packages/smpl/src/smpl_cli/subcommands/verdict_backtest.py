"""`smpl verdict-backtest` — calibrate the gen-QC gate against a human-labeled corpus (vault-2kyt).

Runs the verdict gate (`smpl verdict`) over every entry in a **verdict corpus** and measures it
against the human labels: a confusion matrix (human × gate over keep/listen/cut), precision /
recall per class, the headline **auto-keep accuracy** (of the gate's auto-keeps, how many the
human kept — the ≥ 0.85 target from vault-1apl), auto-cut accuracy, per-axis contribution, and a
threshold-sensitivity sweep (how auto-keep/auto-cut would move if Z_IN_BAND / Z_FAR shifted ±0.5).

This is NOT a pipe stage — it reads a corpus FILE, not stdin frames. It prints a compact
human-readable report to stdout and, with ``--json <path>``, also writes the full JSON report.

Typical use:

    smpl verdict-backtest --corpus verdict-corpus/corpus.jsonl --stats ~/.smpl/stats/kick.json
    smpl verdict-backtest --corpus corpus.jsonl --stats-dir ~/.smpl/stats     # multi-role
    smpl verdict-backtest --corpus corpus.jsonl --stats-dir ~/.smpl/stats --format json

The metrics live in `smpl_analysis.backtest` (which reuses `smpl_analysis.verdict` → triage, the
ONE scoring core); this module is corpus/profile loading + emit glue.
"""

from __future__ import annotations

from .._common import eprint

HELP = "calibrate the verdict gate vs a human-labeled corpus (confusion matrix + auto-keep accuracy)"


def add_arguments(parser):
    parser.add_argument("--corpus", required=True,
                        help="path to a verdict corpus (JSONL or JSON array)")
    parser.add_argument("--stats", default=None,
                        help="a single role stats JSON applied to all entries (from `smpl stats build`)")
    parser.add_argument("--stats-dir", dest="stats_dir", default=None,
                        help="directory of <role>.json profiles; picked per entry role "
                             "(default: ~/.smpl/stats when neither --stats nor --stats-dir given)")
    parser.add_argument("--role", default=None,
                        help="restrict the backtest to entries of this role")
    parser.add_argument("--target", type=float, default=None,
                        help="auto-keep accuracy target (default: 0.85, the vault-1apl gate)")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="also write the full JSON report to this path")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="stdout format (default: text summary; json = full report to stdout)")


def _build_resolver(args):
    """Return a role → (profile, meta) resolver from --stats / --stats-dir (default ~/.smpl/stats)."""
    import json
    from pathlib import Path

    from smpl_analysis import verdict as _verdict

    if args.stats:
        profile, meta = _verdict.load_profile(args.stats)
        return {"profile": profile, "meta": meta}

    stats_dir = Path(args.stats_dir or "~/.smpl/stats").expanduser()
    cache: dict = {}

    def resolver(role):
        if role is None:
            return None
        if role in cache:
            return cache[role]
        path = stats_dir / f"{role}.json"
        if not path.exists():
            cache[role] = None
            return None
        try:
            cache[role] = _verdict.load_profile(str(path))
        except (OSError, json.JSONDecodeError):
            cache[role] = None
        return cache[role]

    return resolver


def run(args) -> int:
    import json

    try:
        from smpl_analysis import backtest as _bt
    except Exception as exc:  # analysis tier not installed
        eprint(f"verdict-backtest: analysis library unavailable: {exc}")
        return 1

    try:
        corpus = _bt.load_corpus(args.corpus)
    except Exception as exc:
        eprint(f"verdict-backtest: cannot read corpus {args.corpus!r}: {exc}")
        return 1
    if not corpus:
        eprint(f"verdict-backtest: corpus {args.corpus!r} is empty")
        return 1

    try:
        resolver = _build_resolver(args)
    except Exception as exc:
        eprint(f"verdict-backtest: cannot load stats: {exc}")
        return 1

    target = args.target if args.target is not None else _bt.TARGET_AUTO_KEEP
    report = _bt.run_backtest(
        corpus, resolver, role=args.role, target_auto_keep=target, corpus_path=args.corpus
    )

    if report["n_scored"] == 0:
        eprint("verdict-backtest: no entries scored "
               f"({report['n_skipped']} skipped — missing profiles or features?)")
        for s in report["skipped"][:10]:
            eprint(f"  skip {s.get('id')}: {s.get('reason')}")
        return 1

    if args.json_out:
        from pathlib import Path

        Path(args.json_out).expanduser().write_text(json.dumps(report, indent=2, sort_keys=True))
        eprint(f"verdict-backtest: wrote JSON report → {args.json_out}")

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_bt.format_report(report))

    # Non-zero exit when below the calibration target, so a CI/routine can gate on it.
    return 0 if report["meets_target"] else 2
