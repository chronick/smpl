# Analysis fan-out — shared agent brief (light tier, P0 depth)

Authoritative references every analysis agent must read:
- Spec (normative): /Users/chronick-mbp/git/vault/music/smplstream/spec.md
- Feature-key registry (USE THESE EXACT KEYS): /Users/chronick-mbp/git/vault/music/smplstream/feature-keys.md
- Pro-audio roadmap (the techniques): /Users/chronick-mbp/git/vault/music/smplstream/research.md

## Foundation API (already built, importable from the synced venv)

`from smplstream import frames as F, cas, error_frame`
- `F.audio_frame(hash, *, sr, ch, dur, role=, of=, lineage=, op=, op_version=, params=, fmt=) -> dict`
- `F.feature_frame(data: dict, *, role=, of=, op=, op_version=, params=, lineage=) -> dict`
- `F.marker_frame(points: list[dict], *, role="onset", of=, op=, op_version=) -> dict`  (each point: {t, sample?, dur?, label?})
- `F.image_frame(hash, *, media="image/png", role="spectrogram", of=, op=, op_version=, meta=) -> dict`
- `F.vector_frame(*, model, dim, dtype="float32", data=None, hash=None, role=, of=) -> dict`  (dim<=64 inline data; dim>64 -> CAS hash, .npy)
- `cas.put_blob(bytes, media) -> hash` ; `cas.put_audio_bytes(wav_bytes) -> hash` ; `cas.get_path(hash) -> Path` ; `cas.read_meta(hash) -> dict`
- `error_frame(code, message, *, of=, op=) -> dict`  (code in: decode_failed, op_failed, resource_exhausted, unsupported, id_collision, not_found)

## CLI subcommand module contract (smpl/subcommands/<name>.py)

```python
HELP = "one-line help"
def add_arguments(parser): ...        # argparse; include selection args if it filters audio
def run(args) -> int: ...             # heavy imports (librosa/matplotlib) INSIDE here
```
Use `from .._common import add_selection_args, selection_mode, read_stdin_frames, emit, eprint`.
Tool contract: passthrough every input frame FIRST (unchanged), then append derived frames.
stderr for humans, stdout for frames. Set of/op/op_version/params on derived frames.

## Library module contract (smpl_analysis/<name>.py)

Pure functions over a resolved audio path / np array returning frame dicts. Heavy imports inside.
Provide `describe_audio_frame(audio_frame, ...)` hooks where the aggregator will call them.

## Hard rules for every agent
- Create ONLY your own NEW files. NEVER edit pyproject.toml, _common.py, dispatch.py,
  registry.py, cat.py, or another tool's files. No shared-file edits.
- Validate with the venv interpreter directly: `/Users/chronick-mbp/git/smpl/.venv/bin/python`
  (and `/Users/chronick-mbp/git/smpl/.venv/bin/pytest <your test>`). DO NOT run any `uv`
  command (parallel uv runs deadlock on the venv lock).
- No new dependencies — librosa, scipy, numpy, soundfile, pyloudnorm, matplotlib are already
  installed. ffmpeg/sox are on PATH for shell-out tools.
- Use the EXACT feature keys from feature-keys.md. Unit-suffix ad-hoc keys.
- Write a test in packages/smpl-analysis/tests/test_<name>.py (new file).
