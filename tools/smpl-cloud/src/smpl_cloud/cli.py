"""`smpl cloud` CLI — a smplstream source tool (provider APIs) + key management.

Generation:  smpl cloud [--provider P] [--prompt PR | --prompt -] [--model M] [--seed N] [--duration S]
Auth:        smpl cloud auth set <provider> <key> | list | rm <provider>

Prompt input — the three spec-required forms (source-tool contract, MUST accept all three):
  1. --prompt "a 90s breakbeat"        (explicit flag — always wins)
  2. --prompt -   /   --prompt --       (raw text on stdin)
  3. a text/role:prompt frame on stdin  (frame stream → consumed, retained with consumed:true)

Two-tier degrade: with no provider SDK installed OR no API key configured, generation emits a
single `error` frame (code `unsupported`) to stdout AND a stderr line with the exact
install/auth command — it never imports a provider SDK at module top and never hangs.
"""

from __future__ import annotations

import io
import sys

from . import backends


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(prog="smpl cloud", description="generate audio via provider APIs")
    p.add_argument("--provider", help="provider (default: $SMPL_CLOUD_PROVIDER or stableaudio)")
    p.add_argument("--prompt", nargs="?", const="-",
                   help="prompt text; '-'/'--' reads raw text from stdin")
    p.add_argument("--model", help="provider model id (default: the provider's default)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=8.0, help="seconds (default: 8.0)")
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--role", default="source")
    return p


def _resolve_prompt(args, stdin_bytes: bytes):
    """Return (prompt, passthrough_frames, consumed_prompt_frame_id). Mirrors smpl gen."""
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


def _emit_unsupported(message: str, hint: str, *, passthrough=None, of=None) -> int:
    """Graceful degrade: an `error` frame (code `unsupported`) to STDOUT + a stderr hint.

    Exit 0 — `unsupported` is a per-frame, non-fatal failure (spec → *Error model*: one
    frame, one failure), not a usage error. The pipe stays composable; downstream sees the
    error frame and can surface the root cause.
    """
    from smplstream import error_frame, ndjson

    out = list(passthrough or [])
    out.append(error_frame("unsupported", message, of=of, op="cloud"))
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    sys.stderr.write(f"smpl cloud: {message}\n  hint: {hint}\n")
    return 0


def _generate(args) -> int:
    import soundfile as sf

    from smplstream import cas, frames as F, ndjson

    stdin_bytes = b"" if sys.stdin.isatty() else sys.stdin.buffer.read()
    prompt, passthrough, consumed_id = _resolve_prompt(args, stdin_bytes)

    if not prompt:
        sys.stderr.write(
            "smpl cloud: no prompt (use --prompt, --prompt -, or a text/prompt frame)\n"
        )
        return 2

    # Resolve the provider. An unknown provider name is itself an `unsupported` degrade.
    try:
        provider = backends.get_provider(args.provider)
    except backends.UnsupportedProvider as exc:
        return _emit_unsupported(str(exc), exc.hint, passthrough=passthrough, of=consumed_id)

    # The whole network path is behind a lazy SDK import + a key lookup. A missing SDK OR a
    # missing key raises UnsupportedProvider — NEVER a raw ImportError — so we route to the
    # single graceful-degrade path. (torch/provider SDKs are imported only inside generate().)
    try:
        samples, sr, model, key = provider.generate(
            prompt, model=args.model, seed=args.seed, duration=args.duration, sr=args.sr
        )
    except backends.UnsupportedProvider as exc:
        return _emit_unsupported(str(exc), exc.hint, passthrough=passthrough, of=consumed_id)

    # --- success path: CAS the audio and emit an audio frame (op:cloud) ---
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
            op="cloud",
            op_version=provider.op_version,
            # cacheable:false — a cloud generation is a non-deterministic remote call (no
            # client-pinned seed/weights guarantee); it MUST NOT be memoized (spec → Memoization).
            params={
                "provider": provider.name,
                "model": model,
                "seed": args.seed,
                "duration": args.duration,
                "prompt": prompt,
                "cacheable": False,
                # NEVER store the key — only a redacted marker for provenance debuggability.
                "key": backends.redact(key),
            },
            lineage=[consumed_id] if consumed_id else None,
            fmt=meta.get("fmt"),
        )
    )
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    return 0


def _auth(argv: list[str]) -> int:
    import json

    action = argv[0] if argv else "list"
    if action == "list":
        for row in backends.auth_list():
            # auth_list already redacts; this prints NO raw key material.
            print(json.dumps(row))
        return 0
    if action == "set" and len(argv) >= 3:
        provider, key = argv[1], argv[2]
        try:
            info = backends.auth_set(provider, key)
        except backends.UnsupportedProvider as exc:
            sys.stderr.write(f"smpl cloud auth: {exc}\n")
            return 2
        # Echo only the redacted result — never the key.
        sys.stderr.write(
            f"smpl cloud auth: stored key for {provider} ({info['key']})\n"
        )
        return 0
    if action == "rm" and len(argv) >= 2:
        ok = backends.auth_rm(argv[1])
        sys.stderr.write(
            f"smpl cloud auth: {'removed ' + argv[1] if ok else 'no stored key for ' + argv[1]}\n"
        )
        return 0 if ok else 1
    sys.stderr.write(
        "usage: smpl cloud auth set <provider> <key> | list | rm <provider>\n"
        f"  providers: {backends.provider_names()}\n"
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "auth":
        return _auth(argv[1:])
    args = _build_parser().parse_args(argv)
    return _generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
