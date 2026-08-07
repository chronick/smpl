"""`smpl envelope` — aligned-hit amplitude envelope as a filter (ticket vault-3tuy).

Passes every input frame through unchanged, then for each selected `audio` frame appends the
aligned-hit envelope `feature` frame(s): one (role ``envelope``) for a one-shot, or two
(roles ``envelope.percussive`` / ``envelope.sub``) for a loop (HPSS + sub-band split). Each
carries the six pinned scalars {peak_db_over_floor, attack_ms_10_90, rise_slope_db_ms, t20_ms,
early_decay_slope, sustain_ratio_150ms}.

Distinct from `smpl env` (the EDIT op that applies a pluck/fade/gate envelope). This is the
analysis op that *measures* the envelope. The DSP lives in `smpl_analysis.envelope`; heavy
imports stay inside `run()`.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "aligned-hit envelope feature frame(s) per audio frame (attack/decay/sustain scalars)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument(
        "--mode", choices=["auto", "oneshot", "loop"], default="auto",
        help="auto (default), oneshot (single-hit passthrough), or loop (HPSS + sub split)",
    )
    parser.add_argument(
        "--bpm", type=float, default=None,
        help="known tempo — sets onset min-spacing to half a 16th note",
    )


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import envelope as _envelope

    rc = 0
    for audio in audios:
        try:
            out.extend(_envelope.envelope_audio_frame(audio, mode=args.mode, bpm=args.bpm))
        except Exception as exc:
            eprint(f"envelope: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="envelope"))
            rc = 1
    emit(out)
    return rc
