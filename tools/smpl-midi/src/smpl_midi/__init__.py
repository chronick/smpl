"""smpl-midi — offline MIDI tools, a smplstream filter pair (kind:midi).

Two verbs, one PATH tool (multicall by argv[0] basename):
  - ``smpl transcribe-midi``  audio frame → midi frame  (audio→MIDI via basic-pitch)
  - ``smpl render-midi``      midi frame  → audio frame  (MIDI→audio via fluidsynth / SC NRT)

Heavy deps (basic-pitch's tensorflow stack, the fluidsynth binary) are isolated in THIS
tool's own venv (two-tier model) and lazy-imported inside ``run()``. With the heavy dep or
the synthesis binary missing, the tool still runs and emits a clean ``error`` frame
(code ``unsupported``) plus a stderr line with the exact install command — it never hangs
and never imports the heavy dep at module top.

Scope: OFFLINE scores/arrangements only (spec → *Out of scope*). NO realtime MIDI / OSC /
live-coding — that is the daemon world, not the content-addressed offline protocol.
"""

__version__ = "0.1.0"
