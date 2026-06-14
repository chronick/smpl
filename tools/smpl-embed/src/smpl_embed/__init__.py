"""smpl-embed — audio embeddings + a FAISS similarity index, smplstream citizens.

`smpl embed` is a *filter*: it reads `audio` frames, runs an embedding model
(MERT / CLAP), and emits `vector` frames (passing every input frame through). Vectors with
`dim > 64` go to CAS as binary `.npy` referenced by hash — NEVER pickle (spec → *Frame
kinds / vector*).

`smpl index` is a *sink*: it builds / queries a FAISS index over emitted vectors.

The heavy ML stack (torch + transformers for MERT/CLAP) and faiss live in THIS tool's own
venv behind extras (two-tier model). They are lazy-imported inside `run()`; a missing
dep/model/binary degrades to a clean `error` frame (code `unsupported`) on stdout plus a
stderr line with the exact install command — never a top-level import of torch.
"""

__version__ = "0.1.0"
