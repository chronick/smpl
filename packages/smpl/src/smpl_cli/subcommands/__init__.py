"""Built-in `smpl` subcommands. One module per command; name `as-wav` ↔ module `as_wav`.

Module contract:
    HELP: str                          # one-line help (read for `smpl --help`)
    def add_arguments(parser): ...      # add argparse args (optional)
    def run(args) -> int: ...           # do the work; heavy imports go INSIDE here
"""
