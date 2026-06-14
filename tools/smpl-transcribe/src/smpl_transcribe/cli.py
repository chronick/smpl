"""`smpl transcribe` CLI — a smplstream *filter* tool (Whisper speech/lyrics) + exporters.

Transcribe:  smpl transcribe [--model M] [--language L] [--role lyrics] [--no-word-timestamps]
Export:      smpl transcribe --format srt|lrc|vtt   (pure-Python, NO heavy dep)
Models:      smpl transcribe models list | install <id> | update <id> | rm <id>

Filter contract (spec → *Tool contract*): read frames from stdin, **pass every input frame
through unchanged** (preserving unknown fields), then emit derived frames. For each inbound
``audio`` frame, resolve its CAS blob to a path, run Whisper, and emit:
  - one ``text`` frame, ``role: lyrics`` (the full transcript), and
  - ``marker`` frames carrying word/segment timestamps (``t`` float-seconds + ``sample`` int
    indexed against the source frame's native ``meta.sr``, per the spec's Units & timebase).

Two-tier discipline: ``whisper``/``torch`` are lazy-imported inside ``run`` (in the backend).
A missing dep/model degrades to a clean ``error`` frame (code ``unsupported``) on stdout PLUS
a stderr line with the exact install command — never a hang, never a top-level torch import.

``--format`` is the export side and needs NO heavy dep: it consumes already-produced
``marker`` + ``text`` frames from stdin and writes SRT / LRC / VTT to stdout (text), passing
the frames through unchanged so the pipe stays composable.
"""

from __future__ import annotations

import sys

from . import backends

EXPORT_FORMATS = ("srt", "lrc", "vtt")

# The exact command to surface on the degrade path (spec → two-tier discipline / stderr hint).
INSTALL_HINT = "uv tool install 'smpl-transcribe[whisper]'"


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="smpl transcribe",
        description="transcribe speech/lyrics from audio frames (Whisper) + srt/lrc/vtt export",
    )
    p.add_argument("--model", help="whisper model id (default: $SMPL_TRANSCRIBE_MODEL or base)")
    p.add_argument("--language", help="force language (e.g. en); default = auto-detect")
    p.add_argument("--role", default="lyrics", help="role for the emitted text frame (default: lyrics)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="whisper sampling temperature; 0.0 pins greedy decoding (default: 0.0)")
    p.add_argument("--no-word-timestamps", dest="word_timestamps", action="store_false",
                   help="emit only segment-level markers, skip per-word timestamps")
    p.add_argument("--format", choices=EXPORT_FORMATS,
                   help="export mode: render existing marker/text frames to srt/lrc/vtt (no model needed)")
    p.set_defaults(word_timestamps=True)
    return p


def _read_stdin_frames() -> list[dict]:
    """Read inbound NDJSON frames (empty list for a tty / empty stdin — a valid empty stream)."""
    import io

    from smplstream import ndjson

    if sys.stdin.isatty():
        return []
    raw = sys.stdin.buffer.read()
    if not raw.strip():
        return []
    return list(ndjson.read_frames(io.BytesIO(raw)))


def _seconds_to_sample(t: float, sr: int) -> int:
    return int(round(t * sr))


# --------------------------------------------------------------------------- #
# Transcription (heavy path — lazy whisper import lives in backends.transcribe)
# --------------------------------------------------------------------------- #

def _markers_from_result(result: dict, sr: int) -> tuple[list[dict], list[dict]]:
    """Split a whisper result into (word_points, segment_points), both sample-indexed.

    Each point is ``{t, sample, dur, label}`` per the spec's marker schema (Units & timebase:
    ``t``/``dur`` float seconds, ``sample`` int against the source native ``sr``).
    """
    word_points: list[dict] = []
    segment_points: list[dict] = []
    for seg in result.get("segments") or []:
        s_start = float(seg.get("start", 0.0))
        s_end = float(seg.get("end", s_start))
        segment_points.append(
            {
                "t": s_start,
                "sample": _seconds_to_sample(s_start, sr),
                "dur": max(0.0, s_end - s_start),
                "label": (seg.get("text") or "").strip(),
            }
        )
        for w in seg.get("words") or []:
            w_start = float(w.get("start", s_start))
            w_end = float(w.get("end", w_start))
            word_points.append(
                {
                    "t": w_start,
                    "sample": _seconds_to_sample(w_start, sr),
                    "dur": max(0.0, w_end - w_start),
                    "label": (w.get("word") or "").strip(),
                }
            )
    return word_points, segment_points


def _transcribe(args, frames: list[dict]) -> int:
    from smplstream import cas, error_frame
    from smplstream import frames as F
    from smplstream import ndjson

    backend = backends.get_backend(args.model)
    op_version = backend.op_version

    # Whisper decoding is non-deterministic on GPU/MPS unless determinism is pinned. We pin
    # greedy decoding at temperature 0.0 by default; if the caller raises temperature the op
    # is no longer reproducible, so we mark cacheable:false in params (spec → Memoization).
    cacheable = args.temperature == 0.0

    out: list[dict] = []
    audio_frames: list[dict] = []
    for f in frames:
        out.append(f)  # passthrough EVERY input frame, unchanged (tool contract)
        if f.get("kind") == "audio" and f.get("hash"):
            audio_frames.append(f)

    if not audio_frames:
        # Empty / no-audio stream is valid: pass through and exit 0 (spec → Out of scope).
        ndjson.write_frames(out)
        sys.stdout.buffer.flush()
        return 0

    failed = False
    for af in audio_frames:
        of_id = af.get("id")
        sr = int((af.get("meta") or {}).get("sr") or 0)
        try:
            audio_path = str(cas.get_path(af["hash"]))
        except FileNotFoundError as exc:
            out.append(error_frame("not_found", f"audio blob unresolved: {exc}",
                                   of=of_id, op="transcribe"))
            failed = True
            continue

        try:
            result = backend.transcribe(
                audio_path,
                language=args.language,
                word_timestamps=args.word_timestamps,
                temperature=args.temperature,
            )
        except ImportError:
            # Two-tier degrade: whisper/torch absent → unsupported frame + stderr install hint.
            out.append(
                error_frame(
                    "unsupported",
                    "whisper backend not installed; "
                    f"install it with: {INSTALL_HINT}",
                    of=of_id,
                    op="transcribe",
                )
            )
            sys.stderr.write(f"smpl transcribe: whisper not installed. Run: {INSTALL_HINT}\n")
            ndjson.write_frames(out)
            sys.stdout.buffer.flush()
            return 1
        except Exception as exc:  # one frame, one failure (spec → Error model)
            out.append(error_frame("op_failed", f"whisper failed: {exc}",
                                   of=of_id, op="transcribe"))
            failed = True
            continue

        if not sr:
            # Pull native rate from CAS meta if the frame omitted it (markers need it).
            meta = cas.read_meta(af["hash"]) or {}
            sr = int(meta.get("sr") or 0)

        word_points, segment_points = _markers_from_result(result, sr)

        params = {
            "model": backend.model_id,
            "language": result.get("language") or args.language,
            "temperature": args.temperature,
            # GPU/MPS nondeterminism: only memoizable when greedy (temperature 0.0) is pinned.
            "cacheable": cacheable,
        }

        out.append(
            F.text_frame(
                (result.get("text") or "").strip(),
                role=args.role,
                of=of_id,
                lineage=[of_id] if of_id else None,
                op="transcribe",
                op_version=op_version,
                params=params,
            )
        )
        if segment_points:
            out.append(
                F.marker_frame(
                    segment_points,
                    role="segment",
                    of=of_id,
                    lineage=[of_id] if of_id else None,
                    op="transcribe",
                    op_version=op_version,
                    params=params,
                )
            )
        if args.word_timestamps and word_points:
            out.append(
                F.marker_frame(
                    word_points,
                    role="word",
                    of=of_id,
                    lineage=[of_id] if of_id else None,
                    op="transcribe",
                    op_version=op_version,
                    params=params,
                )
            )

    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# Exporters (light path — NO heavy dep; render existing marker/text frames)
# --------------------------------------------------------------------------- #

def _fmt_timestamp(t: float, *, comma: bool) -> str:
    """SRT/VTT timestamp ``HH:MM:SS,mmm`` (comma) or ``HH:MM:SS.mmm`` (dot)."""
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _fmt_lrc_timestamp(t: float) -> str:
    """LRC timestamp ``[mm:ss.xx]`` (centiseconds)."""
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"[{m:02d}:{s:02d}.{cs:02d}]"


def _collect_cues(frames: list[dict]) -> list[dict]:
    """Gather timed cues for export from marker frames, preferring segment- over word-level.

    Falls back to a single whole-transcript cue from the text/lyrics frame if no markers are
    present (so ``--format`` still produces something useful from a markerless transcript).
    """
    segs = [f for f in frames if f.get("kind") == "marker" and f.get("role") == "segment"]
    words = [f for f in frames if f.get("kind") == "marker" and f.get("role") == "word"]
    source = segs or words
    cues: list[dict] = []
    for mf in source:
        for pt in mf.get("data") or []:
            label = (pt.get("label") or "").strip()
            if not label:
                continue
            t = float(pt.get("t", 0.0))
            dur = float(pt.get("dur", 0.0) or 0.0)
            cues.append({"t": t, "end": t + dur, "label": label})
    if not cues:
        texts = [f for f in frames if f.get("kind") == "text"
                 and f.get("role") in ("lyrics", "caption")]
        if texts:
            body = (texts[-1].get("data") or "").strip()
            if body:
                cues.append({"t": 0.0, "end": 0.0, "label": body})
    cues.sort(key=lambda c: c["t"])
    return cues


def _render_srt(cues: list[dict]) -> str:
    lines = []
    for i, c in enumerate(cues, 1):
        end = c["end"] if c["end"] > c["t"] else c["t"] + 2.0
        lines.append(str(i))
        lines.append(f"{_fmt_timestamp(c['t'], comma=True)} --> {_fmt_timestamp(end, comma=True)}")
        lines.append(c["label"])
        lines.append("")
    return "\n".join(lines)


def _render_vtt(cues: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for c in cues:
        end = c["end"] if c["end"] > c["t"] else c["t"] + 2.0
        lines.append(f"{_fmt_timestamp(c['t'], comma=False)} --> {_fmt_timestamp(end, comma=False)}")
        lines.append(c["label"])
        lines.append("")
    return "\n".join(lines)


def _render_lrc(cues: list[dict]) -> str:
    return "\n".join(f"{_fmt_lrc_timestamp(c['t'])}{c['label']}" for c in cues) + "\n"


_RENDERERS = {"srt": _render_srt, "vtt": _render_vtt, "lrc": _render_lrc}


def _export(args, frames: list[dict]) -> int:
    """Render existing marker/text frames to subtitles/lyrics. No whisper needed.

    Two modes, set by whether an inbound frame stream is present:
      - **Pure export** (no frames on stdin): write the rendered SRT/LRC/VTT text to stdout.
      - **In-pipe export** (frames on stdin): stdout is reserved for frames (spec rule 5), so
        pass the frames through unchanged on stdout and surface the rendered text on stderr,
        keeping the pipe composable for a downstream `jq`/`smpl` consumer.
    """
    from smplstream import ndjson

    cues = _collect_cues(frames)
    rendered = _RENDERERS[args.format](cues)

    if not frames:
        # Pure export (no inbound frame stream): emit the subtitle/lyric text on stdout.
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        return 0

    # In a frame pipe, stdout is reserved for frames (spec rule 5: stdout is for frames).
    # Pass frames through unchanged and surface the rendered export on stderr so a `jq`/`smpl`
    # consumer downstream still sees a clean frame stream.
    ndjson.write_frames(frames)
    sys.stdout.buffer.flush()
    sys.stderr.write(rendered)
    if not rendered.endswith("\n"):
        sys.stderr.write("\n")
    return 0


# --------------------------------------------------------------------------- #
# Model management
# --------------------------------------------------------------------------- #

def _models(argv: list[str]) -> int:
    import json

    action = argv[0] if argv else "list"
    if action == "list":
        for row in backends.list_models():
            print(json.dumps(row))
        return 0
    if action in ("install", "update") and len(argv) > 1:
        info = backends.install_model(argv[1])
        sys.stderr.write(f"smpl transcribe models: {action}ed {argv[1]} → {info['path']}\n")
        return 0
    if action == "rm" and len(argv) > 1:
        ok = backends.remove_model(argv[1])
        msg = "removed " + argv[1] if ok else "no such model " + argv[1]
        sys.stderr.write(f"smpl transcribe models: {msg}\n")
        return 0 if ok else 1
    sys.stderr.write("usage: smpl transcribe models list|install <id>|update <id>|rm <id>\n")
    return 2


def run(args) -> int:
    frames = _read_stdin_frames()
    if args.format:
        return _export(args, frames)
    return _transcribe(args, frames)


def main(argv: list[str] | None = None) -> int:
    import signal

    # Die cleanly on a closed downstream pipe (spec → Pipe hygiene).
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "models":
        return _models(argv[1:])
    args = _build_parser().parse_args(argv)

    from smplstream.errors import SmplError

    try:
        return run(args)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0
    except KeyboardInterrupt:
        return 130
    except SmplError as exc:
        # Malformed/truncated NDJSON on stdin is a fatal read error (spec → Pipe hygiene):
        # exit non-zero with a clean human message on stderr, never a traceback on stdout.
        sys.stderr.write(f"smpl transcribe: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
