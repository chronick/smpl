"""`smpl cat` (aka `smpl describe`) — describe-as-filter.

Passes every input frame through, then appends derived description frames (`text` +
`feature`, plus `image`/`vector` once the analysis library is installed) for each audio
frame. The richer analysis lives in `smpl_analysis.describe`; this module delegates to it
when present and otherwise emits a light, dependency-free summary (peak/RMS/duration) so
the core is useful before the analysis tier lands.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "describe audio frame(s) as a filter: passthrough + text/feature/image/vector"

OP_VERSION = "cat@1"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--no-image", action="store_true", help="skip spectrogram image frames")


def _minimal_describe(audio: dict) -> list[dict]:
    """Dependency-free fallback: peak/RMS/duration via soundfile + numpy."""
    import numpy as np
    import soundfile as sf

    from smplstream import cas, frames as F

    src = cas.get_path(audio["hash"])
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0

    def dbfs(x: float) -> float:
        return 20.0 * float(np.log10(x)) if x > 0 else -float("inf")

    dur = data.shape[0] / sr if sr else 0.0
    feat = {
        "dur_s": round(dur, 4),
        "sr_hz": sr,
        "ch": data.shape[1],
        "peak_dbfs": round(dbfs(peak), 2),
        "rms_dbfs": round(dbfs(rms), 2),
    }
    summary = (f"{dur:.2f}s · {sr} Hz · {data.shape[1]}ch · "
               f"peak {feat['peak_dbfs']} dBFS · RMS {feat['rms_dbfs']} dBFS")
    return [
        F.feature_frame(feat, role="summary", of=audio["id"], op="cat", op_version=OP_VERSION),
        F.text_frame(summary, role="caption", of=audio["id"], op="cat", op_version=OP_VERSION),
    ]


def run(args) -> int:
    from smplstream import select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    try:
        from smpl_analysis import describe as _describe  # rich path (librosa/MIR)
    except Exception:
        _describe = None

    rc = 0
    for audio in audios:
        try:
            if _describe is not None and hasattr(_describe, "describe_audio_frame"):
                out.extend(_describe.describe_audio_frame(audio, want_image=not args.no_image))
            else:
                out.extend(_minimal_describe(audio))
        except Exception as exc:
            from smplstream import error_frame

            eprint(f"cat: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="cat"))
            rc = 1
    emit(out)
    return rc
