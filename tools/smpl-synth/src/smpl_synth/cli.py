"""`smpl synth` CLI — a smplstream SuperCollider NRT bridge (source + effect) + SynthDef mgmt.

Render:  smpl synth [--synthdef NAME] [--code FILE|-] [--param k=v ...] [--duration S] [--sr R]
SynthDefs: smpl synth defs list | install <name> <file.scd> | rm <name>

SOURCE mode (no upstream audio): renders a SynthDef from scratch → an `audio` frame.
EFFECT mode (an upstream `audio` frame on stdin): resolves that frame's CAS path, feeds it to
`scsynth -N` as the NRT input soundfile, emits a derived `audio` frame with lineage; all other
input frames pass through unchanged (tool contract → *Passthrough*).

Two-tier discipline: SuperCollider is a SYSTEM BINARY, not a pip dep. When sclang/scsynth is
absent the tool emits a clean `unsupported` error frame to stdout AND a stderr line with the
exact install command (`brew install supercollider`) — it NEVER hangs and NEVER imports a heavy
Python dep. (No torch/whisper/etc. exists for this tool; the gate is PATH discovery.)
"""

from __future__ import annotations

import sys

from . import backends


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="smpl synth",
        description="render a SuperCollider SynthDef offline (NRT) — source or effect",
    )
    p.add_argument("--synthdef", help="SynthDef name (registry entry, or the name in --code)")
    p.add_argument("--code", nargs="?", const="-",
                   help="SynthDef .scd source file; '-'/'--' reads raw .scd from stdin")
    p.add_argument("--param", action="append", default=[], metavar="K=V",
                   help="synth arg (repeatable), e.g. --param freq=220 --param amp=0.4")
    p.add_argument("--duration", type=float, default=2.0, help="seconds (default: 2.0)")
    p.add_argument("--sr", type=int, default=44100, help="render sample rate (default: 44100)")
    p.add_argument("--role", help="output role (default: source, or <name>.wet in effect mode)")
    return p


def _parse_params(items: list[str]) -> dict:
    """Parse `--param k=v` pairs into a dict, coercing numeric values (ints/floats stay numeric)."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            sys.stderr.write(f"smpl synth: ignoring malformed --param {item!r} (need k=v)\n")
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _read_stdin_bytes() -> bytes:
    return b"" if sys.stdin.isatty() else sys.stdin.buffer.read()


def _classify_stdin(stdin_bytes: bytes, code_arg) -> tuple[bytes, list[dict]]:
    """Return (raw_code_bytes, input_frames).

    - If --code is '-'/'--', stdin is RAW .scd text → (stdin_bytes, []).
    - Else if stdin looks like an NDJSON frame stream → ([], frames) for effect/passthrough.
    - Else → ([], []) (no usable stdin; source mode with default/registry SynthDef).
    """
    if code_arg in ("-", "--"):
        return stdin_bytes, []
    stripped = stdin_bytes.lstrip()
    if stripped[:1] == b"{":
        import io

        from smplstream import ndjson

        return b"", list(ndjson.read_frames(io.BytesIO(stdin_bytes)))
    return b"", []


def _emit_unsupported(input_frames: list[dict]) -> int:
    """Degrade path: SuperCollider binary absent.

    Emit passthrough frames first, then a clean `unsupported` error frame to STDOUT, and the
    exact install command to STDERR. Exit 0 — an unsupported op is "one frame, one failure"
    (the pipe stays resilient), not a fatal usage error.
    """
    from smplstream import error_frame, ndjson

    missing = backends.missing_binaries()
    msg = (
        f"SuperCollider NRT bridge unavailable: missing {', '.join(missing)} on PATH. "
        f"Install with `{backends.INSTALL_HINT}`."
    )
    sys.stderr.write(f"smpl synth: {msg}\n")
    sys.stderr.write(f"smpl synth: install with: {backends.INSTALL_HINT}\n")

    out = list(input_frames)
    out.append(error_frame("unsupported", msg, op=backends.OP))
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    return 0


def _resolve_input_audio(input_frames: list[dict]):
    """For effect mode: resolve the single (last-wins) upstream audio frame + its CAS path.

    Returns (audio_frame, cas_path) or (None, None) when there is no upstream audio (source
    mode). Surfaces an upstream root-cause error by raising ResolutionError (spec →
    *Error model / Propagation*).
    """
    from smplstream import cas
    from smplstream.select import select

    audio = select(input_frames, kind="audio", predicate=lambda f: bool(f.get("hash")), mode="last")
    if not audio:
        return None, None
    frame = audio[0]
    path = cas.get_path(frame["hash"])
    return frame, str(path)


def _render(args) -> int:
    import io

    from smplstream import cas, error_frame, frames as F, ndjson
    from smplstream.errors import ResolutionError

    stdin_bytes = _read_stdin_bytes()
    raw_code, input_frames = _classify_stdin(stdin_bytes, args.code)

    # --- Two-tier gate: the SuperCollider BINARY (not a pip dep). Lazy PATH discovery only. ---
    if not backends.sc_available():
        return _emit_unsupported(input_frames)

    # Resolve the SynthDef source: --code FILE | --code - (raw stdin) | --synthdef NAME | default.
    code_text = None
    if args.code in ("-", "--"):
        code_text = raw_code.decode("utf-8", "replace")
    elif args.code is not None:
        try:
            with open(args.code, "r", encoding="utf-8") as fh:
                code_text = fh.read()
        except OSError as exc:
            sys.stderr.write(f"smpl synth: cannot read --code {args.code!r}: {exc}\n")
            return 2

    # Effect mode iff an upstream audio frame exists; else source mode.
    try:
        in_audio, in_path = _resolve_input_audio(input_frames)
    except ResolutionError as exc:
        # An upstream op failed (e.g. CUDA-OOM in stems) — surface the root cause, exit non-zero.
        out = list(input_frames)
        out.append(error_frame(getattr(exc, "code", "op_failed"), str(exc),
                               of=getattr(exc, "of", None), op=backends.OP))
        ndjson.write_frames(out)
        sys.stdout.buffer.flush()
        return 1

    effect = in_audio is not None
    try:
        synthdef_source, synth_name = backends.resolve_synthdef(
            code_text, args.synthdef, effect=effect
        )
    except backends.SynthRenderError as exc:
        sys.stderr.write(f"smpl synth: {exc}\n")
        return 2

    params = _parse_params(args.param)

    # --- Render offline (sclang → scsynth -N). Pure offline op → memoizable. ---
    try:
        wav_bytes = backends.render_nrt(
            synthdef_source=synthdef_source,
            synth_name=synth_name,
            params=params,
            duration=args.duration,
            sr=args.sr,
            in_path=in_path,
        )
    except backends.SynthRenderError as exc:
        # One frame, one failure: emit an `op_failed` error frame, keep the pipe resilient.
        out = list(input_frames)
        out.append(
            error_frame("op_failed", str(exc),
                        of=(in_audio.get("id") if in_audio else None), op=backends.OP)
        )
        ndjson.write_frames(out)
        sys.stdout.buffer.flush()
        return 0
    except FileNotFoundError:
        # Binaries vanished between the gate and the render — degrade cleanly.
        return _emit_unsupported(input_frames)

    h = cas.put_audio_bytes(wav_bytes)
    meta = cas.read_meta(h) or {}

    role = args.role or (f"{synth_name}.wet" if effect else "source")
    ov = backends.op_version(synthdef_source)
    out_params = {
        "synthdef": synth_name,
        "sr": args.sr,
        "duration": args.duration,
        "params": params,
        # NRT on CPU is deterministic given (synthdef, params, sr, dur, SC version) — all of
        # which are pinned here or folded into op_version — so this op is memoizable.
        "cacheable": True,
    }
    if effect:
        out_params["input_hash"] = in_audio.get("hash")

    out = list(input_frames)  # passthrough every input frame unchanged, in order, first
    out.append(
        F.audio_frame(
            h,
            sr=meta.get("sr", args.sr),
            ch=meta.get("ch", 2),
            dur=meta.get("dur", args.duration),
            role=role,
            op=backends.OP,
            op_version=ov,
            params=out_params,
            of=(in_audio.get("id") if effect else None),
            lineage=([in_audio.get("id")] if effect else None),
            fmt=meta.get("fmt"),
        )
    )
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    return 0


def _defs(argv: list[str]) -> int:
    import json

    action = argv[0] if argv else "list"
    if action == "list":
        for row in backends.list_synthdefs():
            print(json.dumps(row))
        return 0
    if action == "install" and len(argv) >= 3:
        info = backends.install_synthdef(argv[1], argv[2])
        sys.stderr.write(f"smpl synth defs: installed {argv[1]} → {info['path']}\n")
        return 0
    if action == "rm" and len(argv) >= 2:
        ok = backends.remove_synthdef(argv[1])
        sys.stderr.write(
            f"smpl synth defs: {'removed ' + argv[1] if ok else 'no such SynthDef ' + argv[1]}\n"
        )
        return 0 if ok else 1
    sys.stderr.write("usage: smpl synth defs list|install <name> <file.scd>|rm <name>\n")
    return 2


def main(argv: list[str] | None = None) -> int:
    # SIGPIPE hygiene: a downstream `head` closing the pipe should not raise a Python traceback
    # onto stdout (which would emit a truncated final NDJSON line — a fatal read error). Restore
    # the default so the process dies quietly on a broken pipe.
    try:
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, ValueError, AttributeError):
        pass  # not POSIX (e.g. Windows) — best effort

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "defs":
        return _defs(argv[1:])
    args = _build_parser().parse_args(argv)
    return _render(args)


if __name__ == "__main__":
    raise SystemExit(main())
