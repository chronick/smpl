"""Selection semantics (spec → *Selection semantics*, NORMATIVE).

A role/predicate matching multiple frames defaults to **last-wins** (most-recently-emitted
match, by ``seq`` if present else stream position). ``--all`` returns every match; ``--strict``
errors on >1. Resolving "the single audio frame" follows the same last-wins rule, and MUST
surface a root-cause ``error`` frame instead of a generic not-found when the requested role
failed upstream (spec → *Error model / Propagation*).
"""

from __future__ import annotations

from typing import Callable, Optional

from .errors import ResolutionError


def _matches(frame: dict, role: Optional[str], kind: Optional[str],
             predicate: Optional[Callable[[dict], bool]]) -> bool:
    if kind is not None and frame.get("kind") != kind:
        return False
    if role is not None and frame.get("role") != role:
        return False
    if predicate is not None and not predicate(frame):
        return False
    return True


def _ordered(matches: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
    """Order matches for last-wins. Decide the ordering axis at the SET level, not per frame.

    If EVERY match carries an int ``seq``, order by ``seq`` (survives reordering filters). If
    ANY match lacks ``seq``, fall back to stream position for ALL of them — a per-frame
    seq/no-seq partition would let a no-seq frame jump ahead of a high-seq frame and invert
    last-wins.
    """
    all_seq = all(isinstance(f.get("seq"), int) for _, f in matches)
    if all_seq:
        return sorted(matches, key=lambda t: (t[1]["seq"], t[0]))
    return sorted(matches, key=lambda t: t[0])


def select(
    frames: list[dict],
    *,
    role: Optional[str] = None,
    kind: Optional[str] = None,
    predicate: Optional[Callable[[dict], bool]] = None,
    mode: str = "last",
) -> list[dict]:
    """Select frames by role/kind/predicate. ``mode`` ∈ {"last", "all", "strict"}."""
    matches = _ordered([(i, f) for i, f in enumerate(frames) if _matches(f, role, kind, predicate)])
    if mode == "all":
        return [f for _, f in matches]
    if mode == "strict":
        if len(matches) > 1:
            raise ResolutionError(
                f"--strict: {len(matches)} frames match (role={role!r}, kind={kind!r})"
            )
        return [matches[-1][1]] if matches else []
    # last-wins
    return [matches[-1][1]] if matches else []


def _ancestors(frames: list[dict], frame_id: str) -> set[str]:
    by_id = {f.get("id"): f for f in frames if f.get("id")}
    seen: set[str] = set()
    stack = [frame_id]
    while stack:
        cur = stack.pop()
        f = by_id.get(cur)
        if not f:
            continue
        for parent in ([f["of"]] if f.get("of") else []) + list(f.get("lineage") or []):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def resolve_single_audio(
    frames: list[dict], *, role: Optional[str] = None, strict: bool = False
) -> dict:
    """Resolve the single (last-wins) audio frame, surfacing root-cause errors.

    If no audio frame matches but an ``error`` frame names the requested role/id (or an
    ancestor of it), raise that root cause with its code — not a generic not-found.
    """
    mode = "strict" if strict else "last"
    # A *resolvable* payload must actually carry a hash — a hashless placeholder audio
    # frame (e.g. a role that an upstream op failed to produce) is NOT a payload, so we
    # fall through to surface the root-cause error instead of returning an empty frame.
    audio = select(frames, kind="audio", role=role, predicate=lambda f: bool(f.get("hash")), mode=mode)
    if audio:
        return audio[0]

    # No payload — look for a root-cause error to surface (spec → Propagation).
    errors = [f for f in frames if f.get("kind") == "error"]
    target_ids = {f.get("id") for f in frames if (role is None or f.get("role") == role)}
    for err in errors:
        d = err.get("data") or {}
        of = d.get("of") or err.get("of")
        if of is None:
            continue
        if of in target_ids:
            raise ResolutionError(d.get("message", "upstream op failed"),
                                  code=d.get("code", "op_failed"), of=of)
        # Did an ancestor of a role-matching frame fail?
        for tid in target_ids:
            if tid and of in _ancestors(frames, tid):
                raise ResolutionError(d.get("message", "upstream op failed"),
                                      code=d.get("code", "op_failed"), of=of)
    # A bare role error with no `of` still beats a generic not-found.
    for err in errors:
        d = err.get("data") or {}
        if err.get("role") == role or d.get("role") == role:
            raise ResolutionError(d.get("message", "upstream op failed"),
                                  code=d.get("code", "op_failed"))
    raise ResolutionError(
        f"no resolvable audio frame (role={role!r})" if role else "no resolvable audio frame",
        code="not_found",
    )
