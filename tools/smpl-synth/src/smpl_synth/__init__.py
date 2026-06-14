"""smpl-synth — a SuperCollider NRT bridge, a smplstream *source* + *effect* tool.

`smpl synth` shells out to `sclang`/`scsynth` in NON-realtime (NRT) mode: sclang builds an
OSC score, then `scsynth -N` renders it offline to a soundfile. There is NO live/realtime
server here — that is the daemon/live world and is explicitly out of scope (see spec.md
→ *Out of scope: Realtime MIDI / OSC*). NRT is a pure offline op, so its output is
content-addressable and memoizable.

The "heavy dependency" is the SuperCollider *binary*, not a Python package: it lazy-binds
via PATH discovery (`shutil.which`) inside `run()`, never at import. When sclang/scsynth is
absent the tool emits a clean `unsupported` error frame to stdout + a stderr line with the
exact install command (`brew install supercollider`) — it never hangs and never imports a
heavy Python dep at module top.

Roles:
  - SOURCE: `--synthdef NAME --code FILE` (or `--code -` for stdin) + `--param k=v` →
    an `audio` frame rendered from scratch.
  - EFFECT: with an upstream `audio` frame on stdin, the resolved CAS path is read by the
    NRT render (a SynthDef that reads a buffer) → a derived `audio` frame, lineage attached.

op `sc-nrt`; `op_version` = blake3 of the SynthDef source + the SuperCollider version
fingerprint (a `sclang -version`-style probe) — so a SynthDef edit OR an SC upgrade bumps
the memo key (spec → *Memoization*: weights/impl identity in `op_version`).
"""

__version__ = "0.1.0"
