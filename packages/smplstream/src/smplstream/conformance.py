"""Conformance suite (spec → *normative*; plan.md → Phase 0 risks).

Checks the three invariants the critic flagged — a passthrough-only test misses the other
two:

  1. passthrough     — no input frame silently dropped
  2. lineage closure  — every `of`/`lineage` id resolves (earlier in stream or in CAS),
                        catches consumed-prompt dangling refs
  3. ordering         — `of`/`lineage` targets appear EARLIER in the stream

plus the golden-hash corpus (see tests/test_conformance.py) for canonical-PCM stability.
"""

from __future__ import annotations

from typing import Callable, Optional


def check_passthrough(input_frames: list[dict], output_frames: list[dict]) -> list[str]:
    """Every input frame id must appear in the output (unless legitimately consumed).

    A consumed frame is allowed to be absent ONLY if a passthrough frame retains its id
    with ``consumed: true`` (so the id stays resolvable) — that retained frame still counts
    as present in the output, so the simple id-presence check below is correct.
    """
    out_ids = {f.get("id") for f in output_frames}
    problems = []
    for f in input_frames:
        fid = f.get("id")
        if fid is not None and fid not in out_ids:
            problems.append(f"passthrough: input frame {fid} dropped from output")
    return problems


def check_lineage_closure(
    frames: list[dict], *, cas_resolves: Optional[Callable[[str], bool]] = None
) -> list[str]:
    """Every `of`/`lineage` id must resolve to a frame id in the stream or a CAS blob."""
    ids = {f.get("id") for f in frames if f.get("id")}
    problems = []
    for f in frames:
        refs = []
        if f.get("of"):
            refs.append(f["of"])
        refs.extend(f.get("lineage") or [])
        for ref in refs:
            if ref in ids:
                continue
            if cas_resolves is not None and cas_resolves(ref):
                continue
            problems.append(f"lineage: frame {f.get('id')} references unresolvable {ref}")
    return problems


def check_ordering(frames: list[dict]) -> list[str]:
    """`of`/`lineage` targets must appear earlier in the stream than the referrer."""
    seen: set[str] = set()
    problems = []
    for f in frames:
        refs = []
        if f.get("of"):
            refs.append(f["of"])
        refs.extend(f.get("lineage") or [])
        for ref in refs:
            if ref not in seen:
                # Forward reference (or absent) — only flag refs that exist later in-stream.
                later = any(g.get("id") == ref for g in frames)
                if later:
                    problems.append(
                        f"ordering: frame {f.get('id')} references {ref} that appears later"
                    )
        if f.get("id"):
            seen.add(f["id"])
    return problems


def check_all(
    input_frames: list[dict],
    output_frames: list[dict],
    *,
    cas_resolves: Optional[Callable[[str], bool]] = None,
) -> list[str]:
    """Run all three invariants against a tool's (input, output) frame lists."""
    problems = []
    problems += check_passthrough(input_frames, output_frames)
    problems += check_lineage_closure(output_frames, cas_resolves=cas_resolves)
    problems += check_ordering(output_frames)
    return problems
