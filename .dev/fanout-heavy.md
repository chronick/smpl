# Heavy-generator fan-out — shared agent brief (two-tier separate repos)

Each heavy tool is its OWN uv project under `~/git/smpl/tools/<name>/`, NOT a workspace
member. Installed via its own `uv tool install` → isolated venv → discovered on PATH as
`smpl-<verb>`, reachable through `smpl <verb>`. Glue = PATH discovery only; NO cross-tool
dependency solving. Mirror the existing `tools/smpl-gen/` structure exactly (pyproject with
`[project.scripts]`, `[tool.uv.sources] smplstream = {path=...editable}`, a torch/heavy
`[project.optional-dependencies]` extra, `src/<pkg>/{__init__,cli,backends}.py`).

References: spec.md (Source tools, interop seams, kind:midi), research.md (§5 ML, §3 markers),
plan.md (Generation sources, External engines). Feature keys: feature-keys.md.

## The two-tier discipline (non-negotiable)
- Default install is LIGHT and works WITHOUT the heavy dep: the tool must run, and on a
  missing model/binary/dep emit a clean `error` frame (code `unsupported`) + a stderr line
  with the exact install command — NEVER hang, NEVER import torch at module top.
- Heavy deps (torch, whisper, demucs, transformers, faiss) live behind an extra
  (`smpl-stems[torch]`) and are lazy-imported inside `run()`.
- Model weights are managed under `SMPL_*_HOME`, never a pip dep (ollama-style
  list/install/update/rm registry, like smpl-gen/backends.py).
- API keys (cloud): env-var-first (`SMPL_CLOUD_<PROVIDER>_KEY`, fallback `SMPL_CLOUD_KEY`),
  optional 0600 config via `smpl cloud auth set/list/rm`. NEVER print/log/store keys in
  provenance; redact in params.

## Tools to scaffold

| Dir | PATH name | verb | Role | Heavy dep (extra) | Output frames |
|---|---|---|---|---|---|
| tools/smpl-cloud | smpl-cloud | `smpl cloud` | source (provider APIs) | provider SDKs | audio (op:cloud) |
| tools/smpl-transcribe | smpl-transcribe | `smpl transcribe` | filter | openai-whisper | text(lyrics)+marker(word/seg ts); --format srt/lrc/vtt |
| tools/smpl-stems | smpl-stems | `smpl stems` | filter (1→many) | python-audio-separator / demucs | audio×N (role stem:drums/bass/vocals/other/guitar/piano, op:demucs, op_version w/ weights id) |
| tools/smpl-embed | smpl-embed (+ `smpl index`) | `smpl embed` / `smpl index` | filter / sink | transformers(MERT/CLAP)+faiss | vector (dim>64 → CAS .npy, NEVER pickle), tagged model+version |
| tools/smpl-synth | smpl-synth | `smpl synth` | source + effect | sclang/scsynth binary (NRT) | audio (op:sc-nrt, op_version = SynthDef hash + SC version) |
| tools/smpl-midi | smpl-midi | `smpl transcribe-midi`/`smpl render-midi` | filter | basic-pitch / fluidsynth | midi (kind:midi) ↔ audio |

## Source-tool contract (cloud, synth-as-source) — three prompt forms (MUST accept all)
1. `--prompt "text"` (flag wins)
2. `--prompt -` / `--prompt --` (raw text on stdin)
3. a `text`/role:prompt frame on stdin (consumed, retained with `consumed:true`)
Output: audio frame with op + params (backend/provider, model, seed, resolved prompt),
lineage → consumed prompt frame. Pass non-prompt input frames through.

## Hard rules
- Create ONLY new files under your own tools/<name>/ dir. No edits to the core packages or
  other tools.
- Validate import with `/Users/chronick-mbp/git/smpl/.venv/bin/python` ONLY for smplstream
  import shape; do NOT install torch/whisper (multi-GB). Prove the tool's CLI runs and
  degrades gracefully WITHOUT the heavy dep (the `unsupported` path), and that `--help` works.
- DO NOT run any `uv` command in parallel (lock deadlock). Use the venv interpreter directly.
- Memoization: declare `op_version` incorporating weights identity; non-deterministic GPU
  ops declare cacheable:false.
