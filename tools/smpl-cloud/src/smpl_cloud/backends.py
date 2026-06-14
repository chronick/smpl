"""Cloud providers + key management (plan.md → External engines, env-var-first keys).

This is the install/config surface for `smpl cloud`, the analogue of smpl-gen's backends.py:
a small provider registry plus the key-resolution rules. It holds NO heavy imports — every
provider's network SDK is lazy-imported inside :meth:`CloudProvider.generate`, guarded so a
missing SDK degrades to the `unsupported` path rather than blowing up at module load.

Key resolution (env-var-first, spec/brief):
  1. ``SMPL_CLOUD_<PROVIDER>_KEY``   (per-provider, highest precedence)
  2. ``SMPL_CLOUD_KEY``              (global fallback)
  3. the 0600 config written by ``smpl cloud auth set`` (lowest precedence)

Env ALWAYS overrides the config. Keys are NEVER printed, logged, or stored in provenance —
:func:`redact` masks them everywhere they would otherwise surface.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Optional


class UnsupportedProvider(Exception):
    """A provider cannot run here. Carries the human-facing stderr hint (install/auth) and
    the resolved-but-redacted reason so the CLI can emit one `unsupported` frame + one hint.

    Raised — never the underlying ImportError/missing-key — so callers route to the single
    graceful-degrade path and never leak a stack trace or a key.
    """

    def __init__(self, message: str, *, hint: str):
        super().__init__(message)
        self.hint = hint


# ---------------------------------------------------------------------------
# Config home + secret redaction
# ---------------------------------------------------------------------------

def cloud_home() -> Path:
    return Path(os.environ.get("SMPL_CLOUD_HOME", "~/.smpl/cloud")).expanduser()


def _auth_file() -> Path:
    return cloud_home() / "auth.json"


def redact(key: Optional[str]) -> str:
    """Mask a secret for any human/provenance surface. Never returns the raw key.

    Shows only a length-class hint so logs are debuggable without leaking material.
    """
    if not key:
        return "<unset>"
    return f"<redacted:{len(key)}ch>"


# ---------------------------------------------------------------------------
# Provider registry (the install/config surface)
# ---------------------------------------------------------------------------

class CloudProvider:
    """Base provider. Subclasses set ``name``/``default_model``/``op_version`` and implement
    ``_call`` with a LAZY import of the provider SDK. The network call is fully behind that
    import, so the default (no-SDK) install stays light and degrades cleanly.

    ``cacheable = False``: a cloud generation is a non-deterministic remote call (server-side
    model version, sampling, no client-pinned seed guarantee), so its output MUST NOT be
    memoized as a pure function of inputs (spec → *Memoization*, non-deterministic ops).
    """

    name = "cloud"
    default_model = "default"
    op_version = "cloud:base@1"  # bumped on any behavior change; see note below
    cacheable = False
    sdk_extra = "all"  # the pip extra that provides this provider's SDK
    sdk_module = ""     # the module whose presence proves the SDK is installed

    def env_keys(self) -> list[str]:
        """Env var names checked in precedence order (per-provider, then global)."""
        return [f"SMPL_CLOUD_{self.name.upper()}_KEY", "SMPL_CLOUD_KEY"]

    def install_hint(self) -> str:
        return f"uv tool install 'smpl-cloud[{self.sdk_extra}]'"

    def auth_hint(self) -> str:
        env = self.env_keys()[0]
        return (
            f"set a key: export {env}=... "
            f"(or `smpl cloud auth set {self.name} <key>`)"
        )

    def _require_sdk(self) -> None:
        """Lazy-probe the provider SDK. Raise UnsupportedProvider (NOT ImportError) if absent."""
        if not self.sdk_module:
            return
        try:
            __import__(self.sdk_module)
        except ImportError as exc:  # SDK not installed → the unsupported path
            raise UnsupportedProvider(
                f"provider {self.name!r} SDK not installed",
                hint=self.install_hint(),
            ) from exc

    def _call(self, prompt: str, *, key: str, model: str, seed: int, duration: float, sr: int):
        """Provider-specific network call. Subclasses lazy-import their SDK HERE and return
        ``(samples: np.ndarray float32, sr: int)``. Default providers are placeholders: the
        SDK probe in :meth:`generate` already routed a no-SDK install to `unsupported`, so a
        reachable ``_call`` means the SDK is present but no real client is wired in v1."""
        raise UnsupportedProvider(
            f"provider {self.name!r} network client not implemented in this build",
            hint=self.install_hint(),
        )

    def generate(self, prompt: str, *, model: Optional[str], seed: int, duration: float, sr: int):
        """Resolve a key, ensure the SDK, then call the provider. Raises UnsupportedProvider
        (with a stderr hint) on a missing key or missing SDK — the only graceful-degrade exit.
        Returns ``(samples, sr, resolved_model, key)``; the caller redacts the key."""
        key = resolve_key(self)
        if not key:
            raise UnsupportedProvider(
                f"no API key for provider {self.name!r}",
                hint=self.auth_hint(),
            )
        self._require_sdk()
        resolved_model = model or self.default_model
        samples, out_sr = self._call(
            prompt, key=key, model=resolved_model, seed=seed, duration=duration, sr=sr
        )
        return samples, out_sr, resolved_model, key


class StableAudioProvider(CloudProvider):
    name = "stableaudio"
    default_model = "stable-audio-open-1.0"
    # op_version MUST incorporate the model/weights identity for memoization; cloud calls are
    # cacheable:false regardless (remote nondeterminism), but the model id is recorded so a
    # server-side model change is visible in provenance. Real weights id lands with the client.
    op_version = "cloud:stableaudio@1+stable-audio-open-1.0"
    sdk_extra = "stableaudio"
    sdk_module = "stable_audio_tools"


class ElevenLabsProvider(CloudProvider):
    name = "elevenlabs"
    default_model = "eleven_music_v1"
    op_version = "cloud:elevenlabs@1+eleven_music_v1"
    sdk_extra = "elevenlabs"
    sdk_module = "elevenlabs"


_PROVIDERS = {
    "stableaudio": StableAudioProvider,
    "elevenlabs": ElevenLabsProvider,
}


def default_provider() -> str:
    return os.environ.get("SMPL_CLOUD_PROVIDER", "stableaudio")


def provider_names() -> list[str]:
    return sorted(_PROVIDERS)


def get_provider(name: Optional[str]) -> CloudProvider:
    name = name or default_provider()
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise UnsupportedProvider(
            f"unknown provider {name!r}; known: {provider_names()}",
            hint="pick a known provider with --provider <name>",
        )
    return cls()


# ---------------------------------------------------------------------------
# Key store (env-var-first; 0600 config; env always overrides)
# ---------------------------------------------------------------------------

def resolve_key(provider: CloudProvider) -> Optional[str]:
    """Env-var-first key resolution. Env (per-provider, then global) ALWAYS overrides the
    on-disk config. Returns the raw key (callers MUST redact before logging/provenance)."""
    for env in provider.env_keys():
        val = os.environ.get(env)
        if val:
            return val
    cfg = _read_auth()
    return cfg.get(provider.name)


def _read_auth() -> dict:
    f = _auth_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_auth(cfg: dict) -> None:
    """Write the key config with 0600 perms (owner-only). Created restrictively from the
    start so a key is never briefly world-readable between create and chmod."""
    home = cloud_home()
    home.mkdir(parents=True, exist_ok=True)
    f = _auth_file()
    # Open with O_CREAT|O_WRONLY|O_TRUNC at mode 0600 so the file is never group/other-readable.
    fd = os.open(str(f), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(cfg, fh, indent=2)
    os.chmod(f, stat.S_IRUSR | stat.S_IWUSR)  # belt-and-suspenders 0600


def auth_set(provider: str, key: str) -> dict:
    """Store a key for ``provider`` in the 0600 config. The key is NEVER echoed back."""
    if provider not in _PROVIDERS:
        raise UnsupportedProvider(
            f"unknown provider {provider!r}; known: {provider_names()}",
            hint="pick a known provider",
        )
    cfg = _read_auth()
    cfg[provider] = key
    _write_auth(cfg)
    # Return only redacted/source info — never the key itself.
    return {"provider": provider, "stored": True, "key": redact(key)}


def auth_list() -> list[dict]:
    """List configured providers WITHOUT revealing keys. For each provider report whether a
    key is present and where it resolves from (env override vs config), all redacted."""
    cfg = _read_auth()
    rows = []
    for name in provider_names():
        prov = _PROVIDERS[name]()
        env_src = next((e for e in prov.env_keys() if os.environ.get(e)), None)
        in_cfg = name in cfg
        if env_src:
            source, present = env_src, True
        elif in_cfg:
            source, present = "config", True
        else:
            source, present = None, False
        rows.append({
            "provider": name,
            "key_present": present,
            "source": source,
            "key": redact(resolve_key(prov)) if present else "<unset>",
            "sdk_installed": _sdk_present(prov),
        })
    return rows


def auth_rm(provider: str) -> bool:
    """Remove a provider's key from the config. Does not touch env vars."""
    cfg = _read_auth()
    if provider not in cfg:
        return False
    del cfg[provider]
    _write_auth(cfg)
    return True


def _sdk_present(provider: CloudProvider) -> bool:
    if not provider.sdk_module:
        return True
    try:
        __import__(provider.sdk_module)
        return True
    except ImportError:
        return False
