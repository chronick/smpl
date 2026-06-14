"""Tests for smpl_analysis.describe — the light-tier aggregator.

Verifies that describe_audio_frame aggregates loudness + spectral + qc feature frames,
one mel `image` frame (when want_image), and a synthesized `text` caption — all tagged to
the audio id — and that a failure in one sub-tier yields an `error` frame without aborting
the rest of the aggregation.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from smplstream import cas, frames as F

from smpl_analysis import describe as D


def _tone(sr: int = 22050, dur: float = 1.0, freq: float = 220.0, ch: int = 1) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if ch == 1:
        return sig[:, None]
    return np.column_stack([sig] * ch)


@pytest.fixture()
def cas_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    return tmp_path


@pytest.fixture()
def audio_frame(tmp_path, cas_dir):
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), _tone(), 22050, subtype="FLOAT")
    h = cas.put_audio_file(str(wav))
    meta = cas.read_meta(h) or {}
    return F.audio_frame(
        h, sr=meta.get("sr", 22050), ch=meta.get("ch", 1), dur=meta.get("dur", 1.0),
        role="source", op="read", op_version="read@1",
    )


def _is_png(b: bytes) -> bool:
    return b[:8] == b"\x89PNG\r\n\x1a\n"


def _by_role(frames, role):
    return [f for f in frames if f.get("role") == role]


def test_aggregates_all_tiers_with_image(audio_frame):
    out = D.describe_audio_frame(audio_frame, want_image=True)

    # No passthrough — describe returns only derived frames (caller passes input through).
    assert all(f.get("kind") != "audio" for f in out)

    # One feature frame per light tier, by role.
    assert len(_by_role(out, "loudness")) == 1
    assert len(_by_role(out, "spectral")) == 1
    assert len(_by_role(out, "qc")) == 1

    # Exactly one mel image frame, referencing a real PNG in the CAS.
    images = [f for f in out if f.get("kind") == "image"]
    assert len(images) == 1
    img = images[0]
    assert img["role"] == "spectrogram:mel"
    assert img["of"] == audio_frame["id"]
    assert _is_png(cas.get_path(img["hash"]).read_bytes())

    # Exactly one caption text frame.
    captions = [f for f in out if f.get("kind") == "text" and f.get("role") == "caption"]
    assert len(captions) == 1

    # No errors on a clean tone.
    assert [f for f in out if f.get("kind") == "error"] == []


def test_every_derived_frame_carries_lineage(audio_frame):
    out = D.describe_audio_frame(audio_frame, want_image=True)
    assert out, "expected derived frames"
    for f in out:
        assert f.get("of") == audio_frame["id"], f
    # And every derived frame is structurally valid.
    for f in out:
        assert F.validate_frame(f) == [], f


def test_caption_mentions_headline_numbers(audio_frame):
    out = D.describe_audio_frame(audio_frame, want_image=False)
    captions = [f for f in out if f.get("kind") == "text" and f.get("role") == "caption"]
    assert len(captions) == 1
    text = captions[0]["data"]
    assert isinstance(text, str) and text
    # Duration, sample rate, channels, and the loudness/timbre headline labels are present.
    assert "Hz" in text and "ch" in text
    assert "LUFS" in text and "true-peak" in text
    assert "brightness" in text and "flatness" in text
    # The caption op is the aggregator, tagged as describe@1.
    assert captions[0]["op"] == "describe"
    assert captions[0]["op_version"] == D.OP_VERSION


def test_no_image_skips_image_frame(audio_frame):
    out = D.describe_audio_frame(audio_frame, want_image=False)
    assert [f for f in out if f.get("kind") == "image"] == []
    # The three feature tiers and the caption still come through.
    assert len(_by_role(out, "loudness")) == 1
    assert len(_by_role(out, "spectral")) == 1
    assert len(_by_role(out, "qc")) == 1
    assert len([f for f in out if f.get("role") == "caption"]) == 1


def test_resilient_to_sub_tier_failure(audio_frame, monkeypatch):
    """A raising sub-analysis yields an op_failed error frame and the rest still run."""
    from smpl_analysis import spectral as SP

    def _boom(*a, **k):
        raise RuntimeError("synthetic spectral failure")

    monkeypatch.setattr(SP, "spectral_audio_frame", _boom)

    out = D.describe_audio_frame(audio_frame, want_image=False)

    errors = [f for f in out if f.get("kind") == "error"]
    assert len(errors) == 1
    err = errors[0]
    assert err["data"]["code"] == "op_failed"
    assert "spectral" in err["data"]["message"]
    assert err["data"]["of"] == audio_frame["id"]

    # The other tiers + caption survived the one failure.
    assert len(_by_role(out, "loudness")) == 1
    assert len(_by_role(out, "qc")) == 1
    assert len([f for f in out if f.get("role") == "caption"]) == 1
    # The spectral feature frame is absent (it failed), but describe didn't abort.
    assert _by_role(out, "spectral") == []


def test_synthesize_caption_handles_missing_features(audio_frame):
    """Caption synthesis renders n/a for absent tiers rather than raising."""
    text = D.synthesize_caption(audio_frame, derived=[])
    assert isinstance(text, str) and text
    assert "n/a" in text  # missing loudness/spectral → n/a placeholders
