"""`smpl describe-all` — the light-tier aggregator as a filter.

A thin shell over `smpl_analysis.describe.describe_audio_frame`: for each selected audio
frame it appends the whole light analysis tier in one pass — loudness + spectral + QC
`feature`/`marker` frames, one mel-spectrogram `image` frame, and a synthesized `text`
caption summarizing the headline numbers.

This is the dedicated entrypoint for the aggregator op (`op: describe`). `smpl cat` /
`smpl describe` ALSO delegate to the same library function, but fall back to a light
dependency-free summary when the analysis tier isn't installed; this command always runs
the full aggregator and errors per-frame (never silently degrades) when it can't. All work
lives in `smpl_analysis.describe`; this module just does selection + passthrough + emit.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "aggregate the light analysis tier: loudness+spectral+qc features + mel image + caption"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--no-image", action="store_true", help="skip the mel spectrogram image frame")


def run(args) -> int:
    from smplstream import error_frame, select as S

    from smpl_analysis import describe as D

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    rc = 0
    for audio in audios:
        try:
            out.extend(D.describe_audio_frame(audio, want_image=not args.no_image))
        except Exception as exc:  # the aggregator is resilient, but guard the whole call too
            eprint(f"describe-all: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="describe"))
            rc = 1

    emit(out)
    return rc
