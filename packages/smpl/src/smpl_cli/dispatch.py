"""The multicall dispatcher + git-style external subcommand discovery.

- Invoked as ``smpl`` → subcommand is ``argv[1]``.
- Invoked as ``smpl-cat`` (a console-script shim) → ``argv[0]`` basename gives the
  subcommand (BusyBox multicall, the Python way).
- A built-in subcommand (module under ``subcommands/``) is dispatched in-process.
- A non-built-in ``smpl foo`` execs ``smpl-foo`` on PATH (the extension seam for heavy
  PATH-discovered tools). Built-ins are resolved BEFORE PATH, so our own shims never recurse.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
from typing import Optional

from . import __version__, registry


def _install_sigpipe() -> None:
    # Die cleanly on a closed downstream pipe instead of dumping a BrokenPipe traceback.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass


def _resolve_invocation(argv: list[str]) -> tuple[Optional[str], list[str]]:
    """Return ``(subcommand, remaining_args)`` from argv, honoring multicall shims."""
    prog = os.path.basename(argv[0]) if argv else "smpl"
    if prog.startswith("smpl-") and prog != "smpl":
        return prog[len("smpl-"):], argv[1:]
    rest = argv[1:]
    if not rest:
        return None, []
    return rest[0], rest[1:]


def _top_level_help() -> int:
    builtins = registry.list_builtin_names()
    print("smpl — composable, content-addressed audio-analysis toolchain\n")
    print("usage: smpl <command> [args]\n")
    print("built-in commands:")
    width = max((len(n) for n in builtins), default=0)
    for name in builtins:
        mod = registry.load(name)
        help_text = getattr(mod, "HELP", "") if mod else ""
        print(f"  {name.ljust(width)}  {help_text}")
    print("\nexternal commands: `smpl <x>` execs `smpl-<x>` on PATH when not built in")
    print("  (e.g. smpl gen / smpl cloud / smpl transcribe / smpl stems / smpl synth)")
    print("\n  smpl --version   show version")
    return 0


def _exec_external(name: str, args: list[str]) -> int:
    exe = shutil.which(f"smpl-{name}")
    if exe is None:
        sys.stderr.write(
            f"smpl: unknown command {name!r}\n"
            f"      not a built-in, and no `smpl-{name}` found on PATH.\n"
            f"      Run `smpl --help` for built-ins, or install the tool that provides it.\n"
        )
        return 127
    # Replace this process — the external tool owns stdio from here.
    os.execv(exe, [exe, *args])
    return 127  # unreachable on success


def main(argv: Optional[list[str]] = None) -> int:
    _install_sigpipe()
    argv = list(sys.argv if argv is None else argv)
    subcommand, args = _resolve_invocation(argv)

    if subcommand in (None, "help", "-h", "--help"):
        return _top_level_help()
    if subcommand in ("--version", "-V", "version"):
        print(f"smpl {__version__}")
        return 0

    try:
        mod = registry.load(subcommand)
    except Exception as exc:  # a broken built-in module shouldn't masquerade as "unknown"
        sys.stderr.write(f"smpl {subcommand}: failed to load: {exc}\n")
        return 1

    if mod is None:
        return _exec_external(subcommand, args)

    parser = argparse.ArgumentParser(
        prog=f"smpl {subcommand}", description=getattr(mod, "HELP", None)
    )
    if hasattr(mod, "add_arguments"):
        mod.add_arguments(parser)
    ns = parser.parse_args(args)

    from smplstream.errors import ResolutionError, SmplError

    try:
        return int(mod.run(ns) or 0)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0
    except KeyboardInterrupt:
        return 130
    except ResolutionError as exc:
        sys.stderr.write(f"smpl {subcommand}: {exc} [{getattr(exc, 'code', 'error')}]\n")
        return 1
    except SmplError as exc:
        sys.stderr.write(f"smpl {subcommand}: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
