"""Embedding backends + a thin model registry, and the FAISS index manager.

Backends (MERT / CLAP) are a customizable install surface. The model store path + default
model resolve from env vars (`SMPL_EMBED_HOME`, `SMPL_EMBED_MODEL`). Heavy deps
(`torch` + `transformers`, and `faiss`) are isolated in THIS tool's own venv behind extras
and are **lazy-imported inside `embed()` / index ops** — never at module top — so the
default install runs light. A missing dep/model raises :class:`UnsupportedError`, which the
CLI renders as a clean `unsupported` error frame + a stderr install hint (two-tier model).

Weights are managed under `SMPL_EMBED_HOME` (ollama-style list / install / update / rm),
never as a pip dependency. `op_version` incorporates the **weights identity** (registry
id + revision, or the on-disk weights blake3) so a model upgrade invalidates the memo
cache instead of silently serving stale vectors (spec → *Memoization*).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


class UnsupportedError(Exception):
    """A required heavy dep / model / binary is missing.

    Carries the exact install command so the CLI can emit a clean `unsupported` error frame
    (stdout) plus a human-facing stderr install hint, and never hang or import torch at top.
    """

    def __init__(self, message: str, *, install_hint: str):
        super().__init__(message)
        self.install_hint = install_hint


def embed_home() -> Path:
    return Path(os.environ.get("SMPL_EMBED_HOME", "~/.smpl/embed")).expanduser()


def default_model() -> str:
    return os.environ.get("SMPL_EMBED_MODEL", "mert-v1-95m")


# Hugging Face repo ids for the supported encoders. The pinned `revision` is part of the
# weights identity (and thus `op_version`) so a re-pin re-keys the cache.
_MODEL_CATALOG = {
    "mert-v1-95m": {
        "kind": "mert",
        "hf_repo": "m-a-p/MERT-v1-95M",
        "revision": "main",
        "dim": 768,
        "sr": 24000,
    },
    "mert-v1-330m": {
        "kind": "mert",
        "hf_repo": "m-a-p/MERT-v1-330M",
        "revision": "main",
        "dim": 1024,
        "sr": 24000,
    },
    "clap": {
        "kind": "clap",
        "hf_repo": "laion/clap-htsat-unfused",
        "revision": "main",
        "dim": 512,
        "sr": 48000,
    },
}

_INSTALL_HINT = (
    "uv tool install 'smpl-embed[torch]' && smpl embed models install <model-id>"
)
_FAISS_HINT = "uv tool install 'smpl-embed[faiss]'"


def model_catalog() -> dict:
    return _MODEL_CATALOG


def resolve_model(model_id: Optional[str]) -> tuple[str, dict]:
    """Resolve a model id to its catalog spec (raises UnsupportedError for unknown ids)."""
    model_id = model_id or default_model()
    spec = _MODEL_CATALOG.get(model_id)
    if spec is None:
        raise UnsupportedError(
            f"unknown embedding model {model_id!r}; known: {sorted(_MODEL_CATALOG)}",
            install_hint=_INSTALL_HINT,
        )
    return model_id, spec


def _weights_identity(model_id: str, spec: dict) -> str:
    """Weights identity for `op_version` (spec → *Memoization*, ML weights rule).

    Prefer the blake3 of the on-disk weights if the model is materialized; otherwise fall
    back to the pinned registry `repo@revision`. Either way a model upgrade re-keys the memo
    cache instead of silently serving stale vectors from the old weights.
    """
    info = _registry().get(model_id)
    if info and info.get("weights_blake3"):
        return f"blake3:{info['weights_blake3']}"
    return f"{spec['hf_repo']}@{spec.get('revision', 'main')}"


def op_version(model_id: str, spec: dict) -> str:
    """`op_version` = op tag + model id + weights identity (NOT just the friendly name)."""
    return f"embed:{spec['kind']}:{model_id}:{_weights_identity(model_id, spec)}"


# ---------------------------------------------------------------------------
# Embedding (heavy path — lazy-imported, guarded → UnsupportedError)
# ---------------------------------------------------------------------------


def embed(pcm_f32, sr: int, *, model_id: Optional[str] = None):
    """Embed mono float32 PCM → a 1-D float32 numpy vector.

    Lazy-imports torch + transformers INSIDE this call (never at module top). On a missing
    heavy dep / model weights it raises :class:`UnsupportedError` (the CLI turns that into a
    clean `unsupported` error frame + stderr install hint) — the light default install path.

    NOTE: this is a GPU/MPS-capable ML op. Embeddings are *cacheable: true* only because we
    run on CPU with no nondeterministic kernels and deterministic pooling (mean over the time
    axis); if a GPU/MPS path with nondeterministic ops is ever added, that path MUST set
    cacheable:false in its emitted params (spec → *Memoization*, non-deterministic ops).
    """
    import numpy as np

    model_id, spec = resolve_model(model_id)

    try:  # pragma: no cover - heavy dep, not installed in the light default venv
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        from transformers import AutoModel, AutoProcessor  # noqa: F401
    except ImportError as exc:
        raise UnsupportedError(
            f"embedding model {model_id!r} needs torch + transformers (not installed): {exc}",
            install_hint=_INSTALL_HINT,
        ) from exc

    # pragma: no cover below — exercised only with the heavy extra + downloaded weights.
    import torch  # noqa: F811
    import torchaudio  # noqa: F811
    from transformers import AutoModel, AutoProcessor  # noqa: F811

    target_sr = spec["sr"]
    audio = torch.from_numpy(np.asarray(pcm_f32, dtype=np.float32))
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    try:
        processor = AutoProcessor.from_pretrained(
            spec["hf_repo"], revision=spec.get("revision", "main"), trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            spec["hf_repo"], revision=spec.get("revision", "main"), trust_remote_code=True
        )
    except Exception as exc:  # download/weights problem → unsupported, not a crash
        raise UnsupportedError(
            f"could not load weights for {model_id!r} ({spec['hf_repo']}): {exc}",
            install_hint=_INSTALL_HINT,
        ) from exc

    model.eval()
    with torch.no_grad():
        inputs = processor(audio.numpy(), sampling_rate=target_sr, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        # Mean-pool the last hidden state over the time axis → a single deterministic vector.
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs.hidden_states[-1]
        vec = hidden.mean(dim=1).squeeze(0).to(torch.float32).cpu().numpy()
    return np.asarray(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model registry (ollama-style: list / install / update / rm) under SMPL_EMBED_HOME
# ---------------------------------------------------------------------------


def _registry_file() -> Path:
    return embed_home() / "models.json"


def _registry() -> dict:
    f = _registry_file()
    return json.loads(f.read_text()) if f.exists() else {}


def list_models() -> list[dict]:
    installed = _registry()
    rows = []
    for mid, spec in _MODEL_CATALOG.items():
        info = installed.get(mid)
        rows.append(
            {
                "id": mid,
                "kind": spec["kind"],
                "hf_repo": spec["hf_repo"],
                "revision": spec.get("revision", "main"),
                "dim": spec["dim"],
                "installed": bool(info),
                "op_version": op_version(mid, spec),
                **({"path": info["path"]} if info and info.get("path") else {}),
            }
        )
    return rows


def install_model(model_id: str) -> dict:
    """Register a model as installed. Real weight download lands with the heavy extra; the
    registry + path management (and weights-identity capture) is the v1 surface."""
    model_id, spec = resolve_model(model_id)
    embed_home().mkdir(parents=True, exist_ok=True)
    f = _registry_file()
    reg = _registry()
    path = embed_home() / model_id
    entry: dict = {
        "kind": spec["kind"],
        "hf_repo": spec["hf_repo"],
        "revision": spec.get("revision", "main"),
        "path": str(path),
    }
    # If weights are materialized on disk, capture their blake3 so op_version pins identity.
    wfile = path / "model.safetensors"
    if wfile.exists():
        entry["weights_blake3"] = hashlib.blake2b(wfile.read_bytes(), digest_size=16).hexdigest()
    reg[model_id] = entry
    f.write_text(json.dumps(reg, indent=2))
    return entry


def remove_model(model_id: str) -> bool:
    f = _registry_file()
    if not f.exists():
        return False
    reg = json.loads(f.read_text())
    if model_id not in reg:
        return False
    del reg[model_id]
    f.write_text(json.dumps(reg, indent=2))
    return True


# ---------------------------------------------------------------------------
# FAISS index manager (heavy path — lazy-imported faiss, guarded → UnsupportedError)
# ---------------------------------------------------------------------------


def index_home() -> Path:
    return Path(os.environ.get("SMPL_INDEX_HOME", "~/.smpl/index")).expanduser()


def _index_paths(name: str) -> tuple[Path, Path]:
    safe = "".join(c for c in name if c.isalnum() or c in "-_.") or "default"
    base = index_home() / safe
    return base.with_suffix(".faiss"), base.with_suffix(".ids.json")


def _require_faiss():
    try:  # pragma: no cover - heavy dep, not installed in the light default venv
        import faiss  # noqa: F401
    except ImportError as exc:
        raise UnsupportedError(
            f"`smpl index` needs faiss (not installed): {exc}", install_hint=_FAISS_HINT
        ) from exc
    import faiss  # noqa: F811

    return faiss


def index_build(name: str, vectors, ids: list[str]) -> dict:
    """Build / overwrite a FAISS index `name` over `vectors` (N×dim float32), keyed by ids.

    Lazy-imports faiss; a missing faiss raises :class:`UnsupportedError` (the CLI emits an
    `unsupported` frame + stderr hint). Uses inner-product over L2-normalized vectors, so the
    score is cosine similarity.
    """
    import numpy as np

    faiss = _require_faiss()  # pragma: no cover - requires the faiss extra
    mat = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if mat.ndim != 2 or mat.shape[0] == 0:
        raise UnsupportedError("no vectors to index", install_hint=_FAISS_HINT)
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss_path, ids_path = _index_paths(name)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    ids_path.write_text(json.dumps({"ids": ids, "dim": int(mat.shape[1])}))
    return {"name": name, "count": int(mat.shape[0]), "dim": int(mat.shape[1]), "path": str(faiss_path)}


def index_query(name: str, vector, k: int = 10) -> list[dict]:
    """Query FAISS index `name` for the `k` nearest ids to `vector`. → [{id, score}]."""
    import numpy as np

    faiss = _require_faiss()  # pragma: no cover - requires the faiss extra
    faiss_path, ids_path = _index_paths(name)
    if not faiss_path.exists():
        raise UnsupportedError(
            f"no index named {name!r} at {faiss_path}", install_hint=_FAISS_HINT
        )
    index = faiss.read_index(str(faiss_path))
    ids = json.loads(ids_path.read_text())["ids"]
    q = np.ascontiguousarray(np.asarray([vector], dtype=np.float32))
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, min(k, len(ids)))
    out = []
    for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
        if 0 <= i < len(ids):
            out.append({"id": ids[i], "score": float(score)})
    return out
