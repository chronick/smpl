"""`smpl describe` — alias of `smpl cat` (describe-as-filter). The motivating pipe uses
`smpl describe`; the plan names the op `smpl cat`. They are the same handler."""

from __future__ import annotations

from . import cat

HELP = "alias of `smpl cat` — describe audio frame(s) as a filter"

add_arguments = cat.add_arguments
run = cat.run
