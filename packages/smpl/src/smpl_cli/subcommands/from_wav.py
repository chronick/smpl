"""`smpl from-wav` — read raw WAV from stdin, CAS it under a fresh canonical-PCM hash,
emit an `audio` frame (the bridge back IN from sox/ffmpeg).

`--derives-from <id|role>` records provenance in `params.derives_from`. It is deliberately
NOT a `lineage` edge: the raw-WAV bridge swallows the original frames (`as-wav` emits raw
bytes, not frames; `from-wav` emits only the single new audio frame with no passthrough), so
there is no in-stream frame for a `lineage` id to point at — minting one would dangle and
fail lineage closure. The bytes are new audio (the external op changed them), so they
correctly get a new content hash; a re-run of the identical external pipe over the same
input is memoizable on the `from-wav` op going forward (it does NOT dedup against the
pre-detour audio — an opaque op can't be proven equal).
"""

from __future__ import annotations

import sys

from .._common import emit, eprint

HELP = "CAS raw WAV from stdin and emit an audio frame (sox/ffmpeg bridge back)"

OP_VERSION = "from-wav@1"


def add_arguments(parser):
    parser.add_argument("--role", default="from-wav", help="role for the emitted audio frame")
    parser.add_argument("--derives-from", dest="derives_from",
                        help="upstream role/id label, recorded as provenance (params.derives_from)")


def run(args) -> int:
    from smplstream import cas, frames as F

    wav_bytes = sys.stdin.buffer.read()
    if not wav_bytes:
        eprint("from-wav: empty stdin (expected raw WAV bytes)")
        return 1

    h = cas.put_audio_bytes(wav_bytes)
    meta = cas.read_meta(h) or {}

    # Provenance only — never a lineage edge: the bridge has no in-stream frame to close
    # against, so a `lineage` id here would dangle and fail lineage closure.
    params = {"input_hash": h, "via": "raw-wav-bridge"}
    if args.derives_from:
        params["derives_from"] = args.derives_from

    frame = F.audio_frame(
        h,
        sr=meta.get("sr", 0),
        ch=meta.get("ch", 1),
        dur=meta.get("dur", 0.0),
        role=args.role,
        op="from-wav",
        op_version=OP_VERSION,
        params=params,
        lineage=None,
        fmt=meta.get("fmt"),
    )
    emit([frame])
    return 0
