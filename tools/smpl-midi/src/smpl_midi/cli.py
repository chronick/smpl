"""smpl-midi CLI — two offline MIDI verbs over the smplstream wire protocol.

  smpl transcribe-midi   audio frame → midi frame  (notes; .mid blob in CAS + inline events)
  smpl render-midi       midi frame  → audio frame  (fluidsynth -F / SC NRT, offline)
  smpl render-midi soundfonts list|install <id> <path>|rm <id>   (SoundFont registry)

Multicall dispatch: ONE PATH tool, two console-script shims (``smpl-transcribe-midi`` /
``smpl-render-midi``). We pick the verb from ``argv[0]`` basename — the same multicall trick
the core ``smpl`` dispatcher uses to resolve an external subcommand. So ``smpl transcribe-midi``
execs ``smpl-transcribe-midi`` → here, and ``smpl render-midi`` execs ``smpl-render-midi`` → here.

Two-tier discipline (HARD RULE): heavy deps are NEVER imported at module top. basic-pitch
(audio→MIDI) is lazy-imported inside ``_transcribe`` guarded by try/except ImportError; the
fluidsynth binary (MIDI→audio) is probed with ``shutil.which`` at call time. On a missing
dep/binary/SoundFont the tool emits a clean ``error`` frame (code ``unsupported``) to stdout
AND a stderr line with the exact install command — it never hangs, never tracebacks.
"""

from __future__ import annotations

import io
import os
import sys

from . import backends

# Stdout is for frames; the exact install commands surfaced on stderr for the degrade path.
_TRANSCRIBE_INSTALL = "uv tool install 'smpl-midi[transcribe]'"
_RENDER_INSTALL = (
    "brew install fluid-synth  # the renderer binary; "
    "then `smpl render-midi soundfonts install <id> <path-to.sf2>`"
)


# ---------------------------------------------------------------------------------------
# Helpers shared by both verbs
# ---------------------------------------------------------------------------------------

def _eprint(msg: str) -> None:
    sys.stderr.write(msg if msg.endswith("\n") else msg + "\n")


def _read_stdin_frames() -> list[dict]:
    """Read NDJSON frames from stdin (empty list on an interactive tty)."""
    from smplstream import ndjson

    if sys.stdin.isatty():
        return []
    return list(ndjson.read_frames(sys.stdin.buffer))


def _emit(frames: list[dict]) -> None:
    from smplstream import ndjson

    ndjson.write_frames(frames)
    sys.stdout.buffer.flush()


def _unsupported(passthrough: list[dict], message: str, install_hint: str,
                 *, of: str | None = None, op: str | None = None) -> int:
    """Emit passthrough + a clean `unsupported` error frame; print the install hint to stderr.

    Two-tier degrade path: a missing heavy dep / binary / model is NOT a crash. We stay a good
    smplstream citizen — pass inbound frames through unchanged, then an `error`/`unsupported`
    frame on stdout, and the exact install command on stderr. Exit 0 (one-frame-one-failure:
    the failure is in-band, not a fatal usage error).
    """
    from smplstream import error_frame

    out = list(passthrough)
    out.append(error_frame("unsupported", message, of=of, op=op))
    _emit(out)
    _eprint(f"smpl-midi: {message}")
    _eprint(f"          install: {install_hint}")
    return 0


# ---------------------------------------------------------------------------------------
# Verb: transcribe-midi  (audio frame → midi frame)
# ---------------------------------------------------------------------------------------

def _build_transcribe_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="smpl transcribe-midi",
        description="transcribe an audio frame to a MIDI score (kind:midi) via basic-pitch",
    )
    p.add_argument("--role", default=None, help="select the audio frame with this role (last-wins)")
    p.add_argument("--out-role", default="midi", help="role for the emitted midi frame")
    p.add_argument("--onset-threshold", type=float, default=0.5,
                   help="basic-pitch note onset sensitivity (0..1)")
    p.add_argument("--frame-threshold", type=float, default=0.3,
                   help="basic-pitch frame activation threshold (0..1)")
    p.add_argument("--min-note-len", type=float, default=58.0,
                   help="minimum note length in ms")
    return p


def _notes_to_event_list(pretty_midi_obj) -> list[dict]:
    """Flatten a PrettyMIDI object to a compact, in-spec inline event list.

    Spec timebase: timestamps are float seconds (kind:midi events). Each event:
    {pitch, start, end, velocity, program?}. Kept small enough to inline as `data` (under the
    64 KiB ceiling for short clips); the full .mid blob always lands in CAS regardless.
    """
    events: list[dict] = []
    for inst in pretty_midi_obj.instruments:
        for n in inst.notes:
            events.append({
                "pitch": int(n.pitch),
                "start": round(float(n.start), 4),
                "end": round(float(n.end), 4),
                "velocity": int(n.velocity),
                "program": int(inst.program),
                "drum": bool(inst.is_drum),
            })
    events.sort(key=lambda e: (e["start"], e["pitch"]))
    return events


def _transcribe(args, frames: list[dict]) -> int:
    from smplstream import cas, frames as F, ndjson, select as S
    from smplstream.ids import mint_id

    # Resolve the single (last-wins) audio frame to transcribe. resolve_single_audio surfaces
    # an upstream root-cause error (e.g. a failed stem) instead of a generic not-found.
    try:
        audio = S.resolve_single_audio(frames, role=args.role, strict=False)
    except Exception as exc:  # ResolutionError → no audio to work on; surface it, exit non-zero
        _eprint(f"smpl transcribe-midi: {exc}")
        return 1

    # LAZY heavy import — guarded. basic-pitch pulls a multi-GB tensorflow/coreml stack; it is
    # NEVER imported at module top. Missing → clean unsupported degrade.
    try:
        from basic_pitch.inference import predict  # type: ignore
        from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore
    except ImportError:
        return _unsupported(
            frames,
            "audio→MIDI transcription needs the optional `basic-pitch` backend",
            _TRANSCRIBE_INSTALL,
            of=audio.get("id"),
            op="transcribe-midi",
        )

    src = cas.get_path(audio["hash"])  # CAS path → hand a real file to basic-pitch
    try:
        _model_out, midi_obj, _note_events = predict(
            str(src),
            onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold,
            minimum_note_length=args.min_note_len,
        )
    except Exception as exc:  # one frame, one failure — emit op_failed, keep the pipe alive
        from smplstream import error_frame

        _emit(list(frames) + [error_frame("op_failed", f"basic-pitch failed: {exc}",
                                           of=audio.get("id"), op="transcribe-midi")])
        _eprint(f"smpl transcribe-midi: basic-pitch failed: {exc}")
        return 0

    # Serialize to a .mid blob and store it in CAS (kind:midi heavy payload via `hash`).
    buf = io.BytesIO()
    midi_obj.write(buf)
    mid_bytes = buf.getvalue()
    h = cas.put_blob(mid_bytes, "audio/midi")

    events = _notes_to_event_list(midi_obj)
    op_version = backends.transcribe_op_version()  # incorporates basic-pitch weights identity

    # The midi frame: .mid blob in CAS (hash) is canonical; the small event list rides in meta
    # so a downstream `jq`/render can read notes without materializing the blob. (Per spec the
    # frame carries the heavy payload via `hash`; events live in meta, NOT in `data`, so the
    # "hash xor data" rule holds.)
    midi_frame = mint_id({
        "kind": "midi",
        "hash": h,
        "media": "audio/midi",
        "role": args.out_role,
        "of": audio.get("id"),
        "lineage": [audio["id"]] if audio.get("id") else None,
        "op": "transcribe-midi",
        "op_version": op_version,
        "params": {
            "backend": "basic-pitch",
            "onset_threshold": args.onset_threshold,
            "frame_threshold": args.frame_threshold,
            "min_note_len": args.min_note_len,
        },
        "meta": {
            "n_notes": len(events),
            "n_instruments": len(midi_obj.instruments),
            "events": events if len(events) <= 512 else events[:512],  # bound inline meta size
            "events_truncated": len(events) > 512,
        },
    })
    # Strip a None lineage so the frame stays clean.
    if midi_frame.get("lineage") is None:
        midi_frame.pop("lineage", None)

    _emit(list(frames) + [midi_frame])  # passthrough audio + derived midi (lineage closes)
    return 0


# ---------------------------------------------------------------------------------------
# Verb: render-midi  (midi frame → audio frame)
# ---------------------------------------------------------------------------------------

def _build_render_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="smpl render-midi",
        description="render a MIDI score frame (kind:midi) to audio via fluidsynth (offline)",
    )
    p.add_argument("--role", default=None, help="select the midi frame with this role (last-wins)")
    p.add_argument("--out-role", default="source", help="role for the emitted audio frame")
    p.add_argument("--sr", type=int, default=44100, help="render sample rate (Hz)")
    p.add_argument("--gain", type=float, default=0.5, help="fluidsynth master gain (0..10)")
    return p


def _select_midi_frame(frames: list[dict], role: str | None) -> dict | None:
    """Last-wins midi frame carrying a resolvable payload (`hash` to a .mid blob)."""
    from smplstream import select as S

    matches = S.select(frames, kind="midi", role=role,
                        predicate=lambda f: bool(f.get("hash")), mode="last")
    return matches[0] if matches else None


def _render(args, frames: list[dict]) -> int:
    import shutil
    import subprocess
    import tempfile

    import soundfile as sf

    from smplstream import cas, error_frame, frames as F, select as S

    midi_frame = _select_midi_frame(frames, args.role)
    if midi_frame is None:
        # No midi payload — surface an upstream root-cause error if one named this role/ancestor,
        # else a not_found. Reuse resolve_single_audio's error-propagation shape via a manual scan.
        errs = [f for f in frames if f.get("kind") == "error"]
        if errs:
            d = errs[-1].get("data") or {}
            _eprint(f"smpl render-midi: upstream failure: {d.get('message', 'unknown')}")
            return 1
        _eprint(f"smpl render-midi: no resolvable midi frame (role={args.role!r})")
        return 1

    # Probe the renderer binary + SoundFont LAZILY — both are heavy/external, never at import.
    fs_bin = backends.fluidsynth_bin()
    if fs_bin is None:
        return _unsupported(
            frames,
            "MIDI→audio render needs the `fluidsynth` binary on PATH",
            _RENDER_INSTALL,
            of=midi_frame.get("id"),
            op="render-midi",
        )
    soundfont = backends.default_soundfont()
    if soundfont is None:
        return _unsupported(
            frames,
            "MIDI→audio render needs a SoundFont (.sf2) — none registered",
            "set SMPL_MIDI_SOUNDFONT=/path/to.sf2  OR  "
            "smpl render-midi soundfonts install <id> <path-to.sf2>",
            of=midi_frame.get("id"),
            op="render-midi",
        )

    mid_path = cas.get_path(midi_frame["hash"])  # the .mid blob → a real file for fluidsynth

    # fluidsynth -F renders to a WAV file offline (no live server, no realtime). This is the
    # NRT/offline path the spec sanctions for MIDI→audio; the live scsynth/fluidsynth server
    # world is explicitly out of scope.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        out_wav = tf.name
    try:
        cmd = [
            fs_bin, "-ni", "-F", out_wav,
            "-r", str(args.sr), "-g", str(args.gain),
            str(soundfont), str(mid_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0 or not os.path.getsize(out_wav):
            msg = (proc.stderr or proc.stdout or "no output").strip().splitlines()
            detail = msg[-1] if msg else "fluidsynth produced no audio"
            _emit(list(frames) + [error_frame("op_failed", f"fluidsynth render failed: {detail}",
                                               of=midi_frame.get("id"), op="render-midi")])
            _eprint(f"smpl render-midi: fluidsynth failed: {detail}")
            return 0

        with open(out_wav, "rb") as fh:
            wav_bytes = fh.read()
    finally:
        try:
            os.unlink(out_wav)
        except OSError:
            pass

    h = cas.put_audio_bytes(wav_bytes)  # canonical-PCM keyed audio blob
    meta = cas.read_meta(h) or {}
    op_version = backends.render_op_version(soundfont, fs_bin)  # SoundFont id + fluidsynth ver

    audio_frame = F.audio_frame(
        h,
        sr=meta.get("sr", args.sr),
        ch=meta.get("ch", 1),
        dur=meta.get("dur", 0.0),
        role=args.out_role,
        of=midi_frame.get("id"),
        lineage=[midi_frame["id"]] if midi_frame.get("id") else None,
        op="render-midi",
        op_version=op_version,
        params={
            "backend": "fluidsynth",
            "soundfont": backends.soundfont_identity(soundfont),  # identity only — not a path
            "sr": args.sr,
            "gain": args.gain,
        },
        fmt=meta.get("fmt"),
    )
    _emit(list(frames) + [audio_frame])
    return 0


def _soundfonts(argv: list[str]) -> int:
    """SoundFont registry management: list | install <id> <path> [--default] | rm <id>."""
    import json

    action = argv[0] if argv else "list"
    if action == "list":
        for row in backends.list_soundfonts():
            print(json.dumps(row))
        return 0
    if action == "install" and len(argv) >= 3:
        make_default = "--default" in argv[3:]
        entry = backends.install_soundfont(argv[1], argv[2], make_default=make_default)
        _eprint(f"smpl render-midi soundfonts: installed {argv[1]} → {entry['path']}")
        return 0
    if action == "rm" and len(argv) > 1:
        ok = backends.remove_soundfont(argv[1])
        _eprint(f"smpl render-midi soundfonts: {'removed ' + argv[1] if ok else 'no such soundfont ' + argv[1]}")
        return 0 if ok else 1
    _eprint("usage: smpl render-midi soundfonts list|install <id> <path.sf2> [--default]|rm <id>")
    return 2


# ---------------------------------------------------------------------------------------
# Multicall entrypoint
# ---------------------------------------------------------------------------------------

def _verb_from_argv0() -> str | None:
    """Pick the verb from the invoked binary basename (multicall shim).

    `smpl-transcribe-midi` → "transcribe-midi"; `smpl-render-midi` → "render-midi". A direct
    `python -m smpl_midi.cli` or unexpected name returns None → fall back to argv parsing.
    """
    prog = os.path.basename(sys.argv[0]) if sys.argv else ""
    if prog.startswith("smpl-"):
        return prog[len("smpl-"):]
    return None


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    verb = _verb_from_argv0()

    # Fallback / dev path: allow `python -m smpl_midi.cli transcribe-midi …` by peeling the
    # verb off the front of argv when the multicall basename didn't resolve one.
    if verb is None and raw and raw[0] in ("transcribe-midi", "render-midi"):
        verb = raw[0]
        raw = raw[1:]

    if verb == "transcribe-midi":
        frames = _read_stdin_frames()
        args = _build_transcribe_parser().parse_args(raw)
        return _transcribe(args, frames)

    if verb == "render-midi":
        # SoundFont registry subcommand: `smpl render-midi soundfonts …` (no stdin needed).
        if raw and raw[0] == "soundfonts":
            return _soundfonts(raw[1:])
        frames = _read_stdin_frames()
        args = _build_render_parser().parse_args(raw)
        return _render(args, frames)

    _eprint("smpl-midi: unknown verb (expected `smpl transcribe-midi` or `smpl render-midi`)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
