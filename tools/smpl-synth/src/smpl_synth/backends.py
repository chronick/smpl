"""SuperCollider NRT backend + a thin SynthDef registry.

The "backend" here is the SuperCollider toolchain (`sclang` + `scsynth`), a SYSTEM BINARY
discovered on PATH — never a pip dependency. Nothing in this module imports a heavy Python
package; the only "heavy dep" is the external binary, gated by :func:`sc_available`.

NRT render pipeline (spec → *External synthesis/DSP* seam, offline):
  1. sclang evaluates a small driver program that loads the user SynthDef, builds an OSC
     score (`Score`), and writes the score + compiled synthdefs to a temp dir, then invokes
     `scsynth -N <osc-score> <in-soundfile|_> <out-soundfile> <sr> WAV float` to render
     OFFLINE (no audio device, no realtime clock). On modern SC, `Score.recordNRT` does both
     steps; we drive it through sclang so a single `sclang` invocation produces the WAV.
  2. The rendered WAV is returned to the caller, which CASes it and emits an `audio` frame.

Determinism: NRT on the CPU is deterministic given a fixed SynthDef, params, sample rate,
duration, and SC version — so the op is memoizable. `op_version` folds the SynthDef source
hash together with the SC version fingerprint, so either changing busts the cache.

SynthDef sources live under ``SMPL_SYNTH_HOME`` (default ``~/.smpl/synth``) — managed with
an ollama-style ``list / install / rm`` registry, mirroring smpl-gen's model store. They are
.scd text, not binary weights, so "install" just registers a path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# SuperCollider binaries. `sclang` is the language/driver; `scsynth -N` is the offline server.
SCLANG = "sclang"
SCSYNTH = "scsynth"

# The exact install command surfaced on the unsupported path (stderr) and in the error frame.
INSTALL_HINT = "brew install supercollider"

# op id for every frame this tool produces (spec → frame.op). One stable op for source+effect.
OP = "sc-nrt"


def synth_home() -> Path:
    """SynthDef source store (override with ``SMPL_SYNTH_HOME``)."""
    return Path(os.environ.get("SMPL_SYNTH_HOME", "~/.smpl/synth")).expanduser()


def _resolve_binary(env_var: str, name: str) -> Optional[str]:
    """Resolve a binary via an explicit env override (validated) or PATH.

    An env override that does NOT point at an executable file is treated as absent — so a
    stale/typo'd ``SMPL_SYNTH_SCLANG`` degrades cleanly via the `unsupported` path with a
    precise "missing X" message, rather than sliding into a `FileNotFoundError` at render time
    with an empty missing-list.
    """
    override = os.environ.get(env_var)
    if override:
        return override if os.path.isfile(override) and os.access(override, os.X_OK) else None
    return shutil.which(name)


def _sclang_path() -> Optional[str]:
    return _resolve_binary("SMPL_SYNTH_SCLANG", SCLANG)


def _scsynth_path() -> Optional[str]:
    return _resolve_binary("SMPL_SYNTH_SCSYNTH", SCSYNTH)


def sc_available() -> bool:
    """True only when BOTH sclang and scsynth are resolvable on PATH (the NRT bridge needs both)."""
    return _sclang_path() is not None and _scsynth_path() is not None


def missing_binaries() -> list[str]:
    """Which of (sclang, scsynth) are absent — for a precise stderr/error message."""
    missing = []
    if _sclang_path() is None:
        missing.append(SCLANG)
    if _scsynth_path() is None:
        missing.append(SCSYNTH)
    return missing


def sc_version_fingerprint() -> str:
    """A stable fingerprint of the installed SuperCollider, for `op_version`.

    Uses the shared :func:`smplstream.tool_version_fingerprint` (a ``sclang -version``-style
    probe over both binaries). When SC is absent the helper records an ``__unavailable__``
    marker per command and still returns a stable token — so callers never crash on an absent
    tool. We only build an `op_version` on the supported path anyway.
    """
    from smplstream import tool_version_fingerprint

    sclang = _sclang_path() or SCLANG
    scsynth = _scsynth_path() or SCSYNTH
    # `sclang -version` prints the SC version banner; scsynth `-v` prints its build string.
    return tool_version_fingerprint([sclang, "-version"], [scsynth, "-v"])


def op_version(synthdef_source: str) -> str:
    """`op_version` = SynthDef source hash + SC version fingerprint (spec → *Memoization*).

    A SynthDef edit changes the source hash; an SC upgrade changes the version fingerprint;
    either busts the memo key. Format mirrors the ML-op convention
    (``<impl>@<version>:<weights-hash>``) so a reader can see both identities at a glance.
    """
    from smplstream import blob_hash

    src_hash = blob_hash(synthdef_source.encode("utf-8"))  # blake3:<hex>
    return f"sc-nrt@{sc_version_fingerprint()[:16]}:{src_hash}"


# ---------------------------------------------------------------------------
# NRT render driver.
# ---------------------------------------------------------------------------

# A default SynthDef used when the caller supplies no --code. A real, reproducible source so
# the source-mode pipe works out of the box once SC is installed (parity with smpl-gen's
# procedural `synth` backend). A short percussive sine blip with an exp env.
DEFAULT_SYNTHDEF_NAME = "smplPing"
DEFAULT_SYNTHDEF_SOURCE = """
SynthDef(\\smplPing, { |out=0, freq=440, amp=0.5, dur=1.0, pan=0|
    var env = EnvGen.kr(Env.perc(0.005, dur, 1, -4), doneAction: 2);
    var sig = SinOsc.ar(freq) * env * amp;
    Out.ar(out, Pan2.ar(sig, pan));
}).add;
""".strip()

# Effect-mode default: read the input soundfile from a buffer and pass it through (identity),
# a safe scaffold an effect SynthDef overrides. scsynth -N is given the input soundfile so the
# DiskIn/PlayBuf can read it. Kept minimal — the point is to prove the bridge, not ship a DSP lib.
DEFAULT_EFFECT_NAME = "smplThru"
DEFAULT_EFFECT_SOURCE = """
SynthDef(\\smplThru, { |out=0, bufnum=0, amp=1.0|
    var sig = PlayBuf.ar(2, bufnum, BufRateScale.kr(bufnum), doneAction: 2);
    Out.ar(out, sig * amp);
}).add;
""".strip()


def _driver_program(
    *,
    synthdef_source: str,
    synth_name: str,
    params: dict,
    duration: float,
    sr: int,
    out_path: str,
    in_path: Optional[str],
) -> str:
    """Build the sclang driver that compiles the SynthDef, builds a Score, and records NRT.

    The driver is plain sclang text (no string injection of params into code — params are
    passed as a flat arg array so a hostile param value can't break out into code). It uses
    `Score.recordNRT`, which itself shells `scsynth -N`, so a single `sclang` call renders the
    WAV offline. We render to **float32 WAV at the requested sample rate**.
    """
    import json

    # Param array for the synth's \\new message: [\\freq, 440, \\amp, 0.5, ...]. Numbers stay
    # numbers; strings are quoted by sclang's literal array via JSON-ish rendering below.
    arg_pairs = []
    for k, v in params.items():
        arg_pairs.append(f"\\{k}")
        if isinstance(v, bool):
            arg_pairs.append("1" if v else "0")
        elif isinstance(v, (int, float)):
            arg_pairs.append(repr(v))
        else:
            # Quote as an sclang string literal; json.dumps gives a safe escaped double-quoted form.
            arg_pairs.append(json.dumps(str(v)))
    args_literal = "[" + ", ".join(arg_pairs) + "]"

    in_decl = "nil" if in_path is None else json.dumps(in_path)

    # NB: this is a SCAFFOLD driver. It is exercised only when SC is actually installed; the
    # default (light) install never reaches here. It loads the user SynthDef, schedules one
    # \\new at t=0 and a \\c_set end at t=dur, and records NRT to a float32 WAV.
    return f"""
var synthSource, score, oscPath, inFile;
synthSource = {json.dumps(synthdef_source)};
inFile = {in_decl};
// Compile the user/default SynthDef into a SynthDescLib for Score.
synthSource.interpret;
score = [
    [0.0, ['/s_new', "{synth_name}", 1000, 0, 0] ++ {args_literal}],
    [{float(duration)}, ['/c_set', 0, 0]]  // end marker so the score has a tail
];
Score.recordNRT(
    score,
    oscFilePath: nil,
    outputFilePath: {json.dumps(out_path)},
    inputFilePath: inFile,
    sampleRate: {int(sr)},
    headerFormat: "WAV",
    sampleFormat: "float",
    options: ServerOptions.new.numOutputBusChannels_(2),
    duration: {float(duration)}
);
0.exit;
""".strip()


def render_nrt(
    *,
    synthdef_source: str,
    synth_name: str,
    params: dict,
    duration: float,
    sr: int,
    in_path: Optional[str] = None,
    timeout: float = 120.0,
) -> bytes:
    """Render a SynthDef offline via sclang→scsynth NRT and return the WAV bytes.

    Caller MUST have checked :func:`sc_available` first; this raises ``FileNotFoundError`` if
    the binaries vanished between the check and here. Raises ``SynthRenderError`` on a non-zero
    sclang exit or a missing/empty output file (mapped to an ``op_failed`` error frame by the
    CLI — never a silent success).
    """
    import tempfile

    sclang = _sclang_path()
    if sclang is None:
        raise FileNotFoundError("sclang not found on PATH")

    with tempfile.TemporaryDirectory(prefix="smpl-synth-nrt-") as tmp:
        out_path = str(Path(tmp) / "out.wav")
        driver_path = str(Path(tmp) / "driver.scd")
        Path(driver_path).write_text(
            _driver_program(
                synthdef_source=synthdef_source,
                synth_name=synth_name,
                params=params,
                duration=duration,
                sr=sr,
                out_path=out_path,
                in_path=in_path,
            )
        )
        try:
            proc = subprocess.run(
                [sclang, driver_path],
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SynthRenderError(f"sclang NRT render timed out after {timeout}s") from exc

        out_file = Path(out_path)
        if proc.returncode != 0 or not out_file.exists() or out_file.stat().st_size == 0:
            tail = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace")[-800:]
            raise SynthRenderError(
                f"sclang NRT render failed (exit {proc.returncode}); output: {tail!r}"
            )
        return out_file.read_bytes()


class SynthRenderError(RuntimeError):
    """A non-zero sclang exit or a missing/empty NRT output (→ `op_failed` error frame)."""


# ---------------------------------------------------------------------------
# SynthDef source resolution + a minimal registry (ollama-style: list / install / rm).
# ---------------------------------------------------------------------------


def resolve_synthdef(code: Optional[str], synthdef: Optional[str], *, effect: bool) -> tuple[str, str]:
    """Resolve (synthdef_source, synth_name) from --code / --synthdef.

    Precedence:
      1. ``--code FILE``/``--code -`` (raw .scd text or stdin) supplies the source verbatim.
      2. ``--synthdef NAME`` names a SynthDef in the registry (``SMPL_SYNTH_HOME``).
      3. Neither → the built-in default (a real reproducible source so the bridge works OOTB):
         ``smplPing`` as a source, ``smplThru`` as an effect.

    ``--synthdef`` also names which SynthDef the score instantiates; when only ``--code`` is
    given, ``synthdef`` (if present) selects the name, else the default name is used.
    """
    if code is not None:
        source = code
        name = synthdef or (DEFAULT_EFFECT_NAME if effect else DEFAULT_SYNTHDEF_NAME)
        return source, name
    if synthdef is not None:
        reg = _registry()
        if synthdef in reg:
            src = Path(reg[synthdef]["path"]).read_text()
            return src, synthdef
        raise SynthRenderError(
            f"unknown SynthDef {synthdef!r}; install it with `smpl synth defs install {synthdef} <file.scd>`"
        )
    if effect:
        return DEFAULT_EFFECT_SOURCE, DEFAULT_EFFECT_NAME
    return DEFAULT_SYNTHDEF_SOURCE, DEFAULT_SYNTHDEF_NAME


def _registry_file() -> Path:
    return synth_home() / "synthdefs.json"


def _registry() -> dict:
    import json

    f = _registry_file()
    return json.loads(f.read_text()) if f.exists() else {}


def list_synthdefs() -> list[dict]:
    rows = [
        {"name": DEFAULT_SYNTHDEF_NAME, "builtin": True, "role": "source"},
        {"name": DEFAULT_EFFECT_NAME, "builtin": True, "role": "effect"},
    ]
    for name, meta in _registry().items():
        rows.append({"name": name, "builtin": False, **meta})
    return rows


def install_synthdef(name: str, path: str) -> dict:
    """Register an .scd SynthDef source file under ``SMPL_SYNTH_HOME`` (copies the text in)."""
    import json
    import shutil as _shutil

    synth_home().mkdir(parents=True, exist_ok=True)
    dest = synth_home() / f"{name}.scd"
    _shutil.copyfile(path, dest)
    reg = _registry()
    reg[name] = {"path": str(dest)}
    _registry_file().write_text(json.dumps(reg, indent=2))
    return reg[name]


def remove_synthdef(name: str) -> bool:
    import json

    reg = _registry()
    if name not in reg:
        return False
    try:
        Path(reg[name]["path"]).unlink(missing_ok=True)
    except OSError:
        pass
    del reg[name]
    _registry_file().write_text(json.dumps(reg, indent=2))
    return True
