"""Error model (spec → *Error model*).

"One frame, one failure": a failed op on one frame emits an ``error`` frame and the
pipe continues; only fatal/usage errors exit non-zero. ``error.data`` is
``{code, message, of}`` and SHOULD include ``op``; ``code`` is drawn from a fixed enum
so downstream logic switches on a code, never string-matches a message.
"""

from __future__ import annotations

from typing import Optional

# The standard error-code enum (spec → *Error model*).
ERROR_CODES = frozenset(
    {
        "decode_failed",
        "op_failed",
        "resource_exhausted",
        "unsupported",
        "id_collision",
        "not_found",
    }
)


class SmplError(Exception):
    """Base for fatal smplstream errors (usage / integrity / protocol)."""


class IntegrityError(SmplError):
    """A CAS write whose recomputed hash != target hash (bad bytes never land)."""


class PathSafetyError(SmplError):
    """A hash that does not match the strict ``blake3:<64-hex>`` form."""


class ProtocolError(SmplError):
    """A frame that violates a normative structural invariant."""


class ResolutionError(SmplError):
    """A requested role/id could not be resolved; carries an optional root-cause code."""

    def __init__(self, message: str, *, code: str = "not_found", of: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.of = of


def error_frame(
    code: str,
    message: str,
    *,
    of: Optional[str] = None,
    op: Optional[str] = None,
) -> dict:
    """Build a non-fatal ``error`` frame. ``code`` must be in :data:`ERROR_CODES`."""
    if code not in ERROR_CODES:
        raise ValueError(f"unknown error code {code!r}; must be one of {sorted(ERROR_CODES)}")
    from .ids import mint_id

    data: dict = {"code": code, "message": message}
    if of is not None:
        data["of"] = of
    if op is not None:
        data["op"] = op
    frame: dict = {"kind": "error", "data": data}
    if of is not None:
        frame["of"] = of
    if op is not None:
        frame["op"] = op
    return mint_id(frame)
