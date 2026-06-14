"""smpl-cloud — provider-API audio generation, a smplstream *source* tool.

Cloud provider SDKs (Stable Audio, ElevenLabs, …) are isolated behind per-provider extras in
THIS tool's own venv (two-tier model). The default install is LIGHT: with no provider SDK
installed and no API key configured, the tool still runs and degrades to a clean `unsupported`
error frame plus a stderr install/auth hint — it NEVER imports a provider SDK at module top.

API keys are env-var-first (`SMPL_CLOUD_<PROVIDER>_KEY`, fallback `SMPL_CLOUD_KEY`) with an
optional 0600 config managed by `smpl cloud auth set/list/rm`. Keys are NEVER printed, logged,
or written into frame provenance — they are redacted in `params`.
"""

__version__ = "0.1.0"
