"""smpl — the unified CLI for the smplstream toolchain.

`smpl <cmd>` dispatches to a built-in subcommand (modules under ``subcommands/``) or, when
the name isn't built in, execs ``smpl-<cmd>`` on PATH (git-style external discovery — the
extension seam for heavy PATH-discovered tools, no plugin protocol). Each ``smpl-<cmd>``
console-script shim is the SAME entry point, dispatching on ``argv[0]`` (BusyBox multicall).
"""

__version__ = "0.1.0"
