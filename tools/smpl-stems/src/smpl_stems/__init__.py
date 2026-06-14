"""smpl-stems — source separation, a smplstream 1→many *filter* tool.

Consumes one audio frame and emits N audio frames, one per separated stem
(role ``stem:drums|bass|vocals|other|guitar|piano``). The heavy separator
(Demucs via ``python-audio-separator`` → torch) is isolated in THIS tool's own
venv (two-tier model) and lazy-imported inside ``run()``. Without it, the tool
still runs and emits a clean ``unsupported`` error frame + a stderr install hint
rather than importing torch at module top or hanging.
"""

__version__ = "0.1.0"
