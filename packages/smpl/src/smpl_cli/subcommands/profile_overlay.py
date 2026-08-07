"""`smpl profile-overlay` — spectrum-vs-profile overlay PNG (ticket vault-22oy).

Passes every input frame through unchanged, then for each selected `audio` frame appends an
`image` frame (role ``profile-overlay``): the candidate's 1/6-octave spectrum drawn over the
role-profile median with a shaded p10–p90 band and the six band-deltas printed as text.

The profile is read generically from a stats file — no roles are baked into the tool:

    smpl read cand.wav | smpl profile-overlay --stats ~/.smpl/stats/pad.json

Build the spectrum profile the overlay consumes with `smpl octave-spectrum` + `smpl stats build`.
The rendering lives in `smpl_analysis.triage`; this module is selection + emit glue.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "spectrum-vs-profile overlay image frame per audio frame (candidate curve over role median)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--stats", required=True,
                        help="path to a role stats JSON (from `smpl stats build`)")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    try:
        from smpl_analysis import triage as _triage
    except Exception as exc:  # analysis tier not installed
        eprint(f"profile-overlay: analysis library unavailable: {exc}")
        emit(out)
        return 1

    try:
        profile = _triage.load_stats_file(args.stats)
    except Exception as exc:
        eprint(f"profile-overlay: cannot read stats {args.stats!r}: {exc}")
        emit(out)
        return 1

    rc = 0
    for audio in audios:
        try:
            out.extend(_triage.render_profile_overlay(audio, profile))
        except Exception as exc:
            eprint(f"profile-overlay: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="profile-overlay"))
            rc = 1
    emit(out)
    return rc
