"""SC NRT driver test. SKIPPED when SuperCollider isn't installed (two-tier: the binary is a
system dep). Guards the 0-frame regression: the offline server must receive the compiled
SynthDef via `/d_recv` at t=0, and sclang must exit from `recordNRT`'s completion action rather
than immediately (either half missing yields a 0-frame WAV).

    cd tools/smpl-synth && uv run pytest -q
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from smpl_synth import backends

pytestmark = pytest.mark.skipif(not backends.sc_available(), reason="SuperCollider not on PATH")


def test_default_ping_renders_nonempty():
    """The 0-frame regression: a known-good SynthDef must produce actual audio."""
    wav = backends.render_nrt(synthdef_source=backends.DEFAULT_SYNTHDEF_SOURCE,
                              synth_name="smplPing", params={"freq": 330, "dur": 0.8},
                              duration=1.0, sr=44100)
    x, sr = sf.read(io.BytesIO(wav), dtype="float64", always_2d=True)
    assert x.size > 0
    assert sr == 44100
    assert float(np.max(np.abs(x))) > 0.01
