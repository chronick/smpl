"""`smpl gen` CLI — a smplstream source tool + model management.

Generation:  smpl gen [--backend B] [--prompt P | --prompt -] [--seed N] [--duration S]
Models:      smpl gen models list | install <id> | update <id> | rm <id>

Prompt input — the three spec-required forms (source-tool contract):
  1. --prompt "a 4/4 distorted drum loop"     (explicit flag — always wins)
  2. --prompt -   /   --prompt --              (raw text on stdin)
  3. a text/role:prompt frame on stdin         (frame stream → consumed, retained)
"""

from __future__ import annotations

import io
import sys

from . import backends


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(prog="smpl gen", description="generate audio from a prompt")
    p.add_argument("--backend", help="generation backend (default: $SMPL_GEN_BACKEND or synth)")
    p.add_argument("--prompt", nargs="?", const="-",
                   help="prompt text; '-'/'--' reads raw text from stdin")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=2.0, help="seconds (default: 2.0)")
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--role", default="source")
    return p


def _resolve_prompt(args, stdin_bytes: bytes):
    """Return (prompt, passthrough_frames, consumed_prompt_frame_id)."""
    # Form 1: explicit flag with a real value always wins.
    if args.prompt and args.prompt not in ("-", "--"):
        return args.prompt, [], None

    # Form 2: --prompt - / -- → raw text on stdin.
    if args.prompt in ("-", "--"):
        return stdin_bytes.decode("utf-8", "replace").strip(), [], None

    # Form 3: a frame stream on stdin carrying a text/role:prompt frame.
    stripped = stdin_bytes.lstrip()
    if stripped[:1] == b"{":
        from smplstream import ndjson

        frames = list(ndjson.read_frames(io.BytesIO(stdin_bytes)))
        prompt = None
        consumed_id = None
        passthrough = []
        for f in frames:
            if prompt is None and f.get("kind") == "text" and f.get("role") == "prompt":
                prompt = f.get("data")
                consumed_id = f.get("id")
                f["consumed"] = True  # retain so its id stays resolvable (tool contract)
                passthrough.append(f)
            else:
                passthrough.append(f)
        return prompt, passthrough, consumed_id

    return None, [], None


def _generate(args) -> int:
    import soundfile as sf

    from smplstream import cas, error_frame, frames as F, ndjson

    stdin_bytes = b"" if sys.stdin.isatty() else sys.stdin.buffer.read()
    prompt, passthrough, consumed_id = _resolve_prompt(args, stdin_bytes)

    if not prompt:
        sys.stderr.write("smpl gen: no prompt (use --prompt, --prompt -, or a text/prompt frame)\n")
        return 2

    backend = backends.get_backend(args.backend)
    samples, sr = backend.generate(prompt, seed=args.seed, duration=args.duration, sr=args.sr)

    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}

    out = list(passthrough)
    out.append(
        F.audio_frame(
            h,
            sr=meta.get("sr", sr),
            ch=meta.get("ch", 1),
            dur=meta.get("dur", 0.0),
            role=args.role,
            op="gen",
            op_version=getattr(backend, "op_version", f"gen:{backend.name}@1"),
            params={
                "backend": backend.name,
                "model": backend.name,
                "seed": args.seed,
                "duration": args.duration,
                "prompt": prompt,
            },
            lineage=[consumed_id] if consumed_id else None,
            fmt=meta.get("fmt"),
        )
    )
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    return 0


def _models(argv: list[str]) -> int:
    import json

    action = argv[0] if argv else "list"
    if action == "list":
        for row in backends.list_models():
            print(json.dumps(row))
        return 0
    if action in ("install", "update") and len(argv) > 1:
        info = backends.install_model(argv[1])
        sys.stderr.write(f"smpl gen models: {action}ed {argv[1]} → {info['path']}\n")
        return 0
    if action == "rm" and len(argv) > 1:
        ok = backends.remove_model(argv[1])
        sys.stderr.write(f"smpl gen models: {'removed ' + argv[1] if ok else 'no such model ' + argv[1]}\n")
        return 0 if ok else 1
    sys.stderr.write("usage: smpl gen models list|install <id>|update <id>|rm <id>\n")
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "models":
        return _models(argv[1:])
    args = _build_parser().parse_args(argv)
    return _generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
