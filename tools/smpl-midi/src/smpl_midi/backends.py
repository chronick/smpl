"""MIDI backends + a thin model / SoundFont registry (plan.md → External engines).

Two heavy surfaces, both isolated from the core install:

  - **transcribe** (audio→MIDI): basic-pitch's ICASSP-2022 model. The weights ship inside
    the ``basic-pitch`` wheel, so the weights identity is the package version (recorded in
    ``op_version`` per the spec memoization rule — an ML op_version MUST incorporate the
    weights identity, not just a friendly model name, or a model upgrade silently serves
    stale results).

  - **render** (MIDI→audio): the ``fluidsynth`` BINARY plus a SoundFont (``.sf2``). The
    SoundFont *is* the timbre, so the render ``op_version`` incorporates the SoundFont's
    blake3 + the resolved fluidsynth version. SoundFonts are managed under ``SMPL_MIDI_HOME``
    (ollama-style list/install/rm), never a pip dependency.

Nothing here imports basic-pitch or shells out at import time; resolution is lazy so the
default (light) install loads instantly and degrades to ``unsupported`` cleanly.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path


def midi_home() -> Path:
    return Path(os.environ.get("SMPL_MIDI_HOME", "~/.smpl/midi")).expanduser()


def soundfonts_dir() -> Path:
    return midi_home() / "soundfonts"


# ---- transcribe (audio→MIDI) weights identity ----------------------------------------

def basic_pitch_version() -> str | None:
    """Return the installed basic-pitch version, or None if the heavy dep is absent.

    This is the *weights identity* for the transcribe op: basic-pitch's model weights ship
    inside the wheel, so the package version uniquely identifies the model. Probed via
    importlib.metadata (cheap, no torch/tensorflow import).
    """
    try:
        from importlib import metadata

        return metadata.version("basic-pitch")
    except Exception:
        return None


def transcribe_op_version() -> str:
    """op_version for audio→MIDI, incorporating the basic-pitch weights identity.

    Spec (Memoization): for ML ops, op_version MUST incorporate the weights identity. We use
    the basic-pitch package version (which pins the bundled model). ``unknown`` when the heavy
    dep is absent — transcribe can't run in that state anyway, so the value never reaches the
    cache key for a real result.
    """
    return f"transcribe-midi:basic-pitch@{basic_pitch_version() or 'unknown'}"


# ---- render (MIDI→audio) backend resolution ------------------------------------------

def fluidsynth_bin() -> str | None:
    return shutil.which("fluidsynth")


def fluidsynth_version(exe: str) -> str:
    """Resolved fluidsynth version string for the env fingerprint (shell-out op)."""
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10)
        line = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else ""
        return line or "fluidsynth:unknown"
    except Exception:
        return "fluidsynth:unknown"


def _registry_file() -> Path:
    return midi_home() / "soundfonts.json"


def list_soundfonts() -> list[dict]:
    import json

    f = _registry_file()
    reg = json.loads(f.read_text()) if f.exists() else {}
    default_id = reg.get("default", {}).get("id")
    rows = []
    for sid, meta in reg.items():
        if sid == "default":  # synthetic alias entry — surfaced via `default` flag, not a row
            continue
        path = Path(meta.get("path", ""))
        rows.append({
            "id": sid,
            "installed": path.exists(),
            "path": str(path),
            "sha": meta.get("sha"),
            "default": sid == default_id,
        })
    return rows


def default_soundfont() -> Path | None:
    """Resolve the SoundFont to render with.

    Order: ``SMPL_MIDI_SOUNDFONT`` env var → the registry's ``default`` entry → the first
    ``.sf2`` found under ``soundfonts_dir()``. Returns None when none is available (render
    then degrades to ``unsupported`` with an install hint).
    """
    env = os.environ.get("SMPL_MIDI_SOUNDFONT")
    if env and Path(env).expanduser().exists():
        return Path(env).expanduser()

    import json

    f = _registry_file()
    if f.exists():
        reg = json.loads(f.read_text())
        if "default" in reg:
            p = Path(reg["default"]["path"])
            if p.exists():
                return p
    d = soundfonts_dir()
    if d.exists():
        for sf2 in sorted(d.glob("*.sf2")):
            return sf2
    return None


def soundfont_identity(path: Path) -> str:
    """blake3-ish identity of a SoundFont for op_version (the .sf2 IS the render timbre).

    Uses blake2b from the stdlib (no extra dep) over the file bytes — distinct from the
    protocol's blake3 CAS keys, this only needs to be a stable per-file fingerprint so a
    SoundFont swap busts the render cache.
    """
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sf2:blake2b:" + h.hexdigest()


def render_op_version(soundfont: Path, fluidsynth_exe: str) -> str:
    """op_version for MIDI→audio, incorporating SoundFont identity + fluidsynth version.

    Spec (Memoization): op_version bumps on any behavior change; for a synth render the
    SoundFont and the synth binary both change the output, so both enter the version string.
    """
    return (
        f"render-midi:fluidsynth@{fluidsynth_version(fluidsynth_exe)}"
        f"+{soundfont_identity(soundfont)}"
    )


# ---- minimal SoundFont registry (ollama-style: list / install / rm) ------------------

def install_soundfont(sf_id: str, src_path: str, *, make_default: bool = False) -> dict:
    """Register a local ``.sf2`` under SMPL_MIDI_HOME (copies it into the managed dir)."""
    import json

    src = Path(src_path).expanduser()
    if not src.exists():
        raise SystemExit(f"smpl render-midi: no such SoundFont file: {src}")
    soundfonts_dir().mkdir(parents=True, exist_ok=True)
    dest = soundfonts_dir() / f"{sf_id}.sf2"
    dest.write_bytes(src.read_bytes())

    f = _registry_file()
    reg = json.loads(f.read_text()) if f.exists() else {}
    entry = {"id": sf_id, "path": str(dest), "sha": soundfont_identity(dest)}
    reg[sf_id] = entry
    if make_default or "default" not in reg:
        reg["default"] = entry
    midi_home().mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(reg, indent=2))
    return entry


def remove_soundfont(sf_id: str) -> bool:
    import json

    f = _registry_file()
    if not f.exists():
        return False
    reg = json.loads(f.read_text())
    if sf_id not in reg:
        return False
    entry = reg.pop(sf_id)
    if reg.get("default", {}).get("id") == sf_id:
        reg.pop("default", None)
    try:
        Path(entry["path"]).unlink(missing_ok=True)
    except OSError:
        pass
    f.write_text(json.dumps(reg, indent=2))
    return True
