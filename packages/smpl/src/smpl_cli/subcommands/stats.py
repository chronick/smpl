"""`smpl stats build` — role-corpus statistics from a stream (or directory) of samples.

``smpl stats build --role <name>`` drains the `feature` frames on stdin and reduces each
feature key to a robust distribution ``{median, MAD, p10, p50, p90, n}`` (log domain),
emitting a collection-stats `feature` frame and writing ``~/.smpl/stats/<role>.json``.

Typical use is pipe-native — end an analysis pipe with it:

    smpl read kick*.wav | smpl describe-all | smpl stats build --role kick

``--dir <path>`` is a convenience: it ingests every audio file under a directory, runs the
light describe tier on each (no image), and reduces that — so you can build a role reference
without hand-assembling the pipe.

The reduction + caching live in `smpl_analysis.stats`; this module is selection/ingest glue.
"""

from __future__ import annotations

from .._common import emit, eprint, read_stdin_frames

HELP = "build role-corpus stats (median/MAD/p10/p50/p90/n, log domain) from a feature stream"

_AUDIO_EXTS = (".wav", ".flac", ".aiff", ".aif", ".mp3", ".ogg")


def add_arguments(parser):
    parser.add_argument("action", choices=["build"], help="stats action (currently: build)")
    parser.add_argument("--role", required=True, help="role name for the corpus (names the output file)")
    parser.add_argument(
        "--dir", dest="dir", default=None,
        help="ingest audio files under this directory instead of reading frames from stdin",
    )


def _frames_from_dir(path: str) -> list[dict]:
    """Ingest every audio file under `path` and run the light describe tier on each."""
    from pathlib import Path

    from smplstream import cas, frames as F

    from smpl_analysis import describe as D

    root = Path(path).expanduser()
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in _AUDIO_EXTS)
    out: list[dict] = []
    for f in files:
        try:
            h = cas.put_audio_file(str(f))
            meta = cas.read_meta(h) or {}
            af = F.audio_frame(
                h, sr=meta.get("sr", 0), ch=meta.get("ch", 1), dur=meta.get("dur", 0.0),
                role="source",
            )
            out.append(af)
            out.extend(D.describe_audio_frame(af, want_image=False))
        except Exception as exc:
            eprint(f"stats: skipping {f}: {exc}")
    return out


def run(args) -> int:
    from smpl_analysis import stats as _stats

    if args.dir:
        frames = _frames_from_dir(args.dir)
    else:
        frames = read_stdin_frames()

    try:
        derived = _stats.build_stats(frames, role=args.role)
    except Exception as exc:
        eprint(f"stats build: {exc}")
        return 1

    # Emit the source frames through plus the collection-stats frame.
    emit([*frames, *derived])

    d = derived[0]
    p = d.get("params", {})
    eprint(
        f"stats build: role={args.role} n={p.get('n')} "
        f"keys={len(d.get('data', {}))} "
        f"{'(cache hit)' if p.get('cache_hit') else '→ ' + str(p.get('cas_hash', ''))[:20]}"
    )
    return 0
