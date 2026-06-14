# smplstream

The wire protocol for the `smpl` suite: self-describing **NDJSON frames** that
reference **content-addressed bytes** (never the heavy bytes themselves), over a
local CAS, with **canonical-PCM hashing** and pipeline-wide **memoization**.

This package is the durable interop contract. See the repository root `README.md`
and `spec.md` for the normative protocol.
