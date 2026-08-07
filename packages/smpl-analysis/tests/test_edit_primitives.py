"""Tests for the sampler-edit primitives (crop/reverse/pitch/stretch, vault-2s1g).

Self-contained: CAS a synthetic tone, run each op, resolve the wet frame, assert. Runs anywhere
smplstream + numpy/scipy/soundfile are installed (the phase vocoder needs no librosa).

    ~/.silt/venv/bin/python -m pytest packages/smpl-analysis/tests/test_edit_primitives.py -q
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import edit
from smplstream import cas, frames as F

SR = 44_100


def _tone_frame(f0=200.0, dur=1.0):
    t = np.arange(int(SR * dur)) / SR
    x = (0.6 * np.sin(2 * np.pi * f0 * t)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    return F.audio_frame(h, sr=SR, ch=1, dur=dur, role="tone"), x


def _load(frame):
    d, s = sf.read(str(cas.get_path(frame["hash"])), dtype="float64", always_2d=True)
    return d.mean(axis=1), s


def _peak_hz(y, sr):
    Y = np.abs(np.fft.rfft(y))
    return float(np.fft.rfftfreq(len(y), 1 / sr)[np.argmax(Y)])


def test_crop_window():
    fr, _ = _tone_frame(dur=1.0)
    y, sr = _load(edit.apply_crop(fr, start_s=0.2, end_s=0.5))
    assert len(y) / sr == pytest.approx(0.30, abs=0.005)


def test_reverse():
    fr, x = _tone_frame(dur=0.5)
    y, sr = _load(edit.apply_reverse(fr))
    assert np.allclose(y[:200], x[::-1][:200], atol=1e-3)


def test_pitch_up_octave_keeps_duration():
    fr, _ = _tone_frame(f0=200.0, dur=1.0)
    y, sr = _load(edit.apply_pitch(fr, semitones=12.0))
    assert _peak_hz(y, sr) == pytest.approx(400.0, abs=20)
    assert len(y) / sr == pytest.approx(1.0, abs=0.06)


def test_pitch_down_octave():
    fr, _ = _tone_frame(f0=200.0, dur=1.0)
    y, sr = _load(edit.apply_pitch(fr, semitones=-12.0))
    assert _peak_hz(y, sr) == pytest.approx(100.0, abs=12)


def test_stretch_doubles_duration_keeps_pitch():
    fr, _ = _tone_frame(f0=200.0, dur=1.0)
    y, sr = _load(edit.apply_stretch(fr, ratio=2.0))
    assert len(y) / sr == pytest.approx(2.0, abs=0.15)
    assert _peak_hz(y, sr) == pytest.approx(200.0, abs=12)  # pitch unchanged


def test_stretch_bad_ratio():
    fr, _ = _tone_frame()
    with pytest.raises(ValueError):
        edit.apply_stretch(fr, ratio=0.0)
