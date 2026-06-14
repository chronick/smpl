"""smpl-analysis — the analysis library behind the `smpl` CLI's describe/QC/loudness/etc.

Each analysis op is a small module here (loudness, spectral, qc, describe, images, …) that
operates on smplstream frames and returns new `feature`/`marker`/`image`/`vector`/`text`
frames. The thin CLI subcommands in the `smpl` package call into these. Heavy imports
(librosa, matplotlib) stay inside functions so cold pipe stages start fast.
"""

from __future__ import annotations

__all__: list[str] = []
