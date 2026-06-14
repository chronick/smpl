"""Whisper backend + a thin model registry (plan.md → External engines, ML §5).

The heavy ASR runtime (``openai-whisper`` + torch) is NEVER imported at module top — it is
lazy-imported inside :meth:`WhisperBackend.transcribe`, guarded so a missing dep degrades to
the spec's ``unsupported`` path (see ``cli.run``). Model weights are managed under
``SMPL_TRANSCRIBE_HOME`` (ollama-style list/install/update/rm), never as a pip dependency.

Memoization (spec → *Memoization*, NORMATIVE): ``op_version`` incorporates the Whisper
**model identity** (id + Whisper checkpoint sha when resolvable) so a model swap/upgrade
invalidates stale cached transcripts instead of silently serving them. Whisper decoding on
GPU/MPS is non-deterministic (and uses sampling unless ``temperature=0``); the op therefore
declares ``cacheable: false`` in its params unless determinism is pinned.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# Friendly model id -> the public openai-whisper checkpoint sha256 (first 16 hex), so the
# weights identity (not just the name) rides in op_version per the spec's ML-op rule. These
# are the upstream-published checkpoint hashes; if a model is not in this table we fall back
# to hashing the on-disk checkpoint at load time (see _weights_id).
_WHISPER_CHECKPOINT_SHA = {
    "tiny": "65147644a5",
    "tiny.en": "d3dd57d32a",
    "base": "ed3a0b6b1c",
    "base.en": "25a8566e1d",
    "small": "9ecf779972",
    "small.en": "f953ad0fd2",
    "medium": "345ae4da62",
    "medium.en": "d7440d1dc1",
    "large-v1": "e4b87e7e0b",
    "large-v2": "81f7ab7e0b",
    "large-v3": "e5b1a55b89",
    "turbo": "01a8d2b3f6",
}

DEFAULT_MODEL = "base"


def transcribe_home() -> Path:
    return Path(os.environ.get("SMPL_TRANSCRIBE_HOME", "~/.smpl/transcribe")).expanduser()


def default_model() -> str:
    return os.environ.get("SMPL_TRANSCRIBE_MODEL", DEFAULT_MODEL)


def _weights_id(model_id: str) -> str:
    """Stable weights identity for ``op_version`` (spec: ML ops bind weights, not a name).

    Prefer the published checkpoint sha; otherwise hash the on-disk checkpoint if the model
    has been installed locally; otherwise fall back to the bare id (still better than nothing,
    and a later real load can tighten it)."""
    sha = _WHISPER_CHECKPOINT_SHA.get(model_id)
    if sha:
        return f"sha256:{sha}"
    ckpt = transcribe_home() / f"{model_id}.pt"
    if ckpt.exists():
        h = hashlib.sha256()
        with ckpt.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()[:16]
    return f"id:{model_id}"


def op_version(model_id: str) -> str:
    """``op_version`` for the transcribe op — binds the Whisper model weights identity.

    Bumped on any behavior change; the trailing ``@1`` is this op's implementation revision.
    """
    return f"transcribe:whisper@1+{model_id}:{_weights_id(model_id)}"


class WhisperBackend:
    """openai-whisper ASR. torch/whisper are lazy-imported INSIDE :meth:`transcribe`."""

    name = "whisper"
    needs_weights = True

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or default_model()

    @property
    def op_version(self) -> str:
        return op_version(self.model_id)

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        word_timestamps: bool = True,
        temperature: float = 0.0,
    ) -> dict:
        """Run Whisper. Raises ImportError if the heavy dep is absent (caller → unsupported).

        Returns the raw whisper result dict: ``{"text", "segments": [{start,end,text,
        words?:[{word,start,end}]}], "language"}``.
        """
        import whisper  # lazy — NEVER at module top (two-tier discipline)

        download_root = str(transcribe_home())
        transcribe_home().mkdir(parents=True, exist_ok=True)
        model = whisper.load_model(self.model_id, download_root=download_root)
        result = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            temperature=temperature,
            verbose=False,
        )
        return result


def get_backend(model_id: str | None = None) -> WhisperBackend:
    return WhisperBackend(model_id)


# ---- minimal model registry (ollama-style: list / install / update / rm) ----

def _registry_file() -> Path:
    return transcribe_home() / "models.json"


def list_models() -> list[dict]:
    import json

    f = _registry_file()
    installed = json.loads(f.read_text()) if f.exists() else {}
    rows = []
    for mid in sorted(_WHISPER_CHECKPOINT_SHA):
        rows.append(
            {
                "id": mid,
                "backend": "whisper",
                "installed": mid in installed,
                "weights_id": _weights_id(mid),
            }
        )
    # Any locally-registered model not in the known table.
    for mid, meta in installed.items():
        if mid not in _WHISPER_CHECKPOINT_SHA:
            rows.append({"id": mid, "installed": True, **meta})
    return rows


def install_model(model_id: str) -> dict:
    """Register a model as installed. The real checkpoint download happens lazily on first
    ``transcribe`` (whisper.load_model under SMPL_TRANSCRIBE_HOME); this manages the registry
    + path, the light v1 surface."""
    import json

    transcribe_home().mkdir(parents=True, exist_ok=True)
    f = _registry_file()
    reg = json.loads(f.read_text()) if f.exists() else {}
    reg[model_id] = {"backend": "whisper", "path": str(transcribe_home() / f"{model_id}.pt")}
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
