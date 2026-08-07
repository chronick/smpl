"""`smpl vocode` — channel-vocode the selected CARRIER with a MODULATOR (2-audio-input op).

The first two-input op in the `smpl` suite: the MODULATOR's per-band amplitude envelope is
imprinted onto the CARRIER's spectrum, so the output articulates like the modulator but keeps
the carrier's timbre. Art-direction linchpin for WRUM (the hell-pole voice): FLOW spat render =
MODULATOR, deep harsh bass voice = CARRIER.

  carrier  — the last-wins selected `audio` frame from stdin (``--role`` narrows the pick).
  modulator — ``--with <PATH-or-role>``: an existing file path is CAS-ingested as a `source`
              frame AND emitted (lineage closure); otherwise it resolves as a role in the stream.

Passthrough every input frame first, then append the ingested modulator (if a file) and one wet
`audio` frame (role ``<carrier_role>.wet``, ``op: vocode``, ``lineage: [carrier.id, modulator.id]``).
DSP lives in ``smpl_analysis.duo``. Follows the ``normalize`` subcommand structure.

  smpl read bass.wav | smpl vocode --with flow-render.wav --bands 24 | smpl normalize --lufs -12 | smpl write wrum.wav
"""

from __future__ import annotations

import os

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "channel-vocode the selected carrier with a modulator (--with PATH|role); emits <role>.wet"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--with", dest="modulator", required=True,
                        help="modulator: an audio file PATH (ingested + emitted) or a role in the stream")
    parser.add_argument("--bands", type=int, default=24, help="number of log-spaced vocoder bands (default 24)")
    parser.add_argument("--lo", type=float, default=70.0, help="low edge of the band grid, Hz (default 70)")
    parser.add_argument("--hi", type=float, default=11000.0, help="high edge of the band grid, Hz (default 11000)")
    parser.add_argument("--ess", type=float, default=5500.0,
                        help="ess-band highpass cutoff, Hz — modulator HF for consonant clarity (default 5500)")
    parser.add_argument("--ess-mix", type=float, default=0.35,
                        help="ess-band mix as a fraction of carrier RMS (default 0.35; 0 disables)")
    parser.add_argument("--attack", type=float, default=4.0, help="envelope-follower attack, ms (default 4)")
    parser.add_argument("--release", type=float, default=45.0, help="envelope-follower release, ms (default 45)")


def run(args) -> int:
    from smplstream import cas, error_frame, frames as F, select as S

    inframes = read_stdin_frames()
    out = list(inframes)

    # Carrier: last-wins selected audio frame from the stream (--role narrows).
    carriers = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not carriers and inframes:
        carriers = S.select(inframes, kind="audio", mode="last")
    if not carriers:
        eprint("vocode: no carrier audio frame in the stream")
        out.append(error_frame("not_found", "vocode: no carrier audio frame in the stream", op="vocode"))
        emit(out)
        return 1
    carrier = carriers[-1]

    # Modulator: --with is a file PATH (ingest + emit for lineage closure) or a role in the stream.
    modulator = None
    spec = args.modulator
    if os.path.isfile(spec):
        try:
            h = cas.put_audio_file(spec)
            meta = cas.read_meta(h) or {}
            modulator = F.audio_frame(
                h,
                sr=meta.get("sr", 0),
                ch=meta.get("ch", 1),
                dur=meta.get("dur", 0.0),
                role="source",
                op="read",
                op_version="read@1",
                fmt=meta.get("fmt"),
                params={"source": spec},
            )
            out.append(modulator)  # emit the ingested modulator so its lineage is in-stream
        except Exception as exc:
            eprint(f"vocode: failed to ingest modulator {spec!r}: {exc}")
            out.append(error_frame("decode_failed", f"{spec}: {exc}", op="vocode"))
            emit(out)
            return 1
    else:
        mods = S.select(inframes, kind="audio", role=spec, mode="last")
        if not mods:
            eprint(f"vocode: modulator role {spec!r} not found in the stream (and not a file path)")
            out.append(error_frame("not_found",
                                    f"vocode: modulator {spec!r} is neither a file path nor a stream role",
                                    op="vocode"))
            emit(out)
            return 1
        modulator = mods[-1]

    from smpl_analysis import duo

    rc = 0
    try:
        out.append(duo.apply_vocode(
            carrier,
            modulator,
            bands=args.bands,
            lo_hz=args.lo,
            hi_hz=args.hi,
            ess_hz=args.ess,
            ess_mix=args.ess_mix,
            attack_ms=args.attack,
            release_ms=args.release,
        ))
    except Exception as exc:
        eprint(f"vocode: {carrier.get('id')}: {exc}")
        out.append(error_frame("op_failed", str(exc), of=carrier.get("id"), op="vocode"))
        rc = 1
    emit(out)
    return rc
