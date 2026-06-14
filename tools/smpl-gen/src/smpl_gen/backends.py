"""Generation backends + a thin model registry (plan.md → Generation sources).

Backends are a customizable install surface. The model store path + default backend resolve
from env vars (`SMPL_GEN_HOME`, `SMPL_GEN_BACKEND`). The default `synth` backend needs no
weights; heavy backends (musicgen/…) register only when their deps import, and their weights
are managed under `SMPL_GEN_HOME` — never as a pip dependency.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


def gen_home() -> Path:
    return Path(os.environ.get("SMPL_GEN_HOME", "~/.smpl/gen")).expanduser()


def default_backend() -> str:
    return os.environ.get("SMPL_GEN_BACKEND", "synth")


class SynthBackend:
    """Deterministic synthesis from a text prompt — no model weights, no torch.

    Not a music model; a real, reproducible audio source so pipes work end-to-end. The
    prompt seeds pitch/timbre so `op_version` + prompt + seed fully determine the output
    (memoizable, conformance-friendly).
    """

    name = "synth"
    op_version = "gen:synth@1"
    needs_weights = False

    def generate(self, prompt: str, *, seed: int = 0, duration: float = 2.0, sr: int = 44100):
        h = hashlib.blake2b((prompt + f"|seed={seed}").encode(), digest_size=16).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little") ^ (seed & 0xFFFFFFFF))

        n = max(1, int(duration * sr))
        t = np.arange(n, dtype=np.float64) / sr

        # Map prompt bytes → a small chord; keywords nudge timbre.
        root = 55.0 * (2.0 ** ((h[0] % 24) / 12.0))  # A1..~A3
        intervals = [0, 7, 12] if (h[1] % 2) else [0, 3, 7]
        sig = np.zeros(n, dtype=np.float64)
        for k, semi in enumerate(intervals):
            f = root * (2.0 ** (semi / 12.0))
            sig += (0.6 ** k) * np.sin(2 * np.pi * f * t)

        p = prompt.lower()
        if any(w in p for w in ("distort", "noise", "gritty", "harsh")):
            sig += 0.4 * rng.standard_normal(n)
        if any(w in p for w in ("drum", "perc", "kick", "beat", "loop")):
            bpm = 120.0
            period = sr * 60.0 / bpm
            env = np.exp(-3.0 * ((np.arange(n) % period) / period))
            sig *= 0.3 + 0.7 * env

        # Gentle fade to avoid clicks; normalize to -1 dBFS headroom.
        fade = min(n, int(0.01 * sr))
        if fade:
            sig[:fade] *= np.linspace(0, 1, fade)
            sig[-fade:] *= np.linspace(1, 0, fade)
        peak = np.max(np.abs(sig)) or 1.0
        sig = (sig / peak) * 0.891  # ~ -1 dBFS
        return sig.astype(np.float32), sr


def available_backends() -> dict:
    backends = {"synth": SynthBackend()}
    # Heavy backends register only if importable — keeps the default path light.
    try:  # pragma: no cover - optional heavy dep
        import audiocraft  # noqa: F401

        from .musicgen import MusicGenBackend  # type: ignore

        backends["musicgen"] = MusicGenBackend()
    except Exception:
        pass
    return backends


def get_backend(name: str | None):
    name = name or default_backend()
    backends = available_backends()
    if name not in backends:
        raise SystemExit(
            f"smpl gen: backend {name!r} not available. Installed: {sorted(backends)}. "
            f"Heavy backends need `uv tool install 'smpl-gen[torch]'` + "
            f"`smpl gen models install <id>`."
        )
    return backends[name]


# ---- minimal model registry (ollama-style: list / install / update / rm) ----

def _registry_file() -> Path:
    return gen_home() / "models.json"


def list_models() -> list[dict]:
    import json

    f = _registry_file()
    installed = json.loads(f.read_text()) if f.exists() else {}
    rows = [{"backend": "synth", "id": "synth", "installed": True, "size": "0 (procedural)"}]
    for mid, meta in installed.items():
        rows.append({"id": mid, "installed": True, **meta})
    return rows


def install_model(model_id: str) -> dict:
    """Register a model as installed. Real weight download lands with the heavy backends;
    the registry + path management is the v1 surface."""
    import json

    gen_home().mkdir(parents=True, exist_ok=True)
    f = _registry_file()
    reg = json.loads(f.read_text()) if f.exists() else {}
    reg[model_id] = {"backend": model_id.split(":")[0], "path": str(gen_home() / model_id)}
    f.write_text(json.dumps(reg, indent=2))
    return reg[model_id]


def remove_model(model_id: str) -> bool:
    import json

    f = _registry_file()
    if not f.exists():
        return False
    reg = json.loads(f.read_text())
    if model_id not in reg:
        return False
    del reg[model_id]
    f.write_text(json.dumps(reg, indent=2))
    return True
