---
title: "smplstream feature-key registry"
created: 2026-06-14
tags: [music, audio, smplstream, analysis, reference, interop]
description: "The canonical key names, namespaces, and units for smplstream `feature` frames. Shared source of truth so analysis tools don't mint divergent spellings. Versioned with SCHEMA."
---

# smplstream feature-key registry

**The source of truth for `feature`-frame key names** ([spec.md](spec.md) →
*Standards alignment*). Every analysis tool that emits a `feature` frame MUST
use a key registered here; adding a measurement means **adding a row here
first**, then emitting it. This is what stops six tickets from independently
spelling spectral flatness three different ways.

Versioned with the `SCHEMA` constant (a key rename/removal is a `SCHEMA` bump;
adding a row is additive and non-breaking).

## Conventions (from spec → Units & timebase / Standards)

- **Objective MIR features** use the **Essentia / AcousticBrainz** namespaces
  (`lowlevel.*`, `rhythm.*`, `tonal.*`); the unit is implied by the namespaced
  key, and frame-aggregated values carry the `{mean, stdev}` statistic shape.
- **Perceptual / LLM-facing** descriptors use the **AudioCommons** `timbre.*`
  prefix — deliberately **outside** the Essentia namespace (they're perceptual,
  not signal-objective).
- **Domain keys with no Essentia equivalent** (loudness, QC) use a short prefix
  (`loudness.*`, `qc.*`) and, per the spec, **MUST suffix the unit**
  (`_lufs`, `_dbtp`, `_db`, `_dbfs`, `_hz`) so a bare number is never ambiguous.

## Registry

| Key | Namespace | Unit | Stat | Owner |
|---|---|---|---|---|
| `loudness.integrated_lufs` | loudness (unit-suffixed) | LUFS | scalar | vault-3vau |
| `loudness.true_peak_dbtp` | loudness | dBTP | scalar | **vault-3vau (sole owner; QC reuses)** |
| `loudness.max_short_term_lufs` | loudness | LUFS | scalar | vault-3vau |
| `loudness.lra` | loudness | LU | scalar | (P2 dynamics) |
| `lowlevel.spectral_flatness_db` | Essentia | dB (NOT 0–1 ratio) | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_crest` | Essentia | unitless | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_spread` | Essentia | Hz | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_rolloff` | Essentia | Hz | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_contrast` | Essentia | dB | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_slope` | Essentia | unitless | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_skewness` | Essentia | unitless | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_kurtosis` | Essentia | unitless | {mean,stdev} | vault-3uap |
| `qc.clipping.detected` | qc | bool | scalar | vault-1e9a |
| `qc.phase.correlation` | qc | unitless (−1..1) | scalar | vault-1e9a |
| `qc.dc_offset_dbfs` | qc | dBFS | scalar | vault-1e9a |
| `qc.snr_db` | qc | dB | scalar | vault-1e9a |
| `qc.lossy.spectral_cutoff_hz` | qc | Hz | scalar | vault-1e9a |
| `qc.lossy.expected_nyquist_hz` | qc | Hz | scalar | vault-1e9a |
| `qc.lossy.confidence` | qc | 0–1 | scalar | vault-1e9a |
| `timbre.hardness` | AudioCommons (perceptual) | 0–100 | scalar | vault-14ia |
| `timbre.depth` | AudioCommons | 0–100 | scalar | vault-14ia |
| `timbre.brightness` | AudioCommons | 0–100 | scalar | vault-14ia |
| `timbre.roughness` | AudioCommons | 0–100 | scalar | vault-14ia |
| `timbre.warmth` | AudioCommons | 0–100 | scalar | vault-14ia |
| `timbre.sharpness` | AudioCommons | 0–100 | scalar | vault-14ia |
| `timbre.boominess` | AudioCommons | 0–100 | scalar | vault-14ia |
| `timbre.reverb` | AudioCommons | binary (0/1) | scalar | vault-14ia |
| `rhythm.bpm` | Essentia | BPM | scalar | vault-32n3 |
| `rhythm.bpm_confidence` | Essentia | 0–1 | scalar | vault-32n3 |
| `rhythm.bpm_candidates` | Essentia | BPM[] | list | vault-32n3 |
| `rhythm.time_signature` | Essentia | n/d string | scalar | vault-32n3 |
| `tonal.key_key` | Essentia | pitch class | scalar | vault-379o |
| `tonal.key_scale` | Essentia | major/minor | scalar | vault-379o |
| `tonal.tuning_frequency` | Essentia | Hz | scalar | vault-379o |
| `fingerprint.chromaprint` | fingerprint | id (int-array/base64) | scalar | vault-2xro |

## Ownership notes (avoid double-emission)

- **True-peak overs** (`loudness.true_peak_dbtp` + over-location markers) are
  owned by **vault-3vau** (loudness tier). The QC ticket (vault-1e9a) **reuses**
  that frame for its clipping pass/fail rather than recomputing under a `qc.*`
  key. One measurement, one owner.
- `timbre.*` (AudioCommons, perceptual) is intentionally separate from
  `lowlevel.*` (Essentia, objective) even where they sound similar
  (`timbre.sharpness` ≠ any `lowlevel.*` — the former is the perceptual 0–100
  descriptor, MoSQITo/AudioCommons-derived).

## Status

The Essentia-namespaced rows (`lowlevel.*`, `rhythm.*`, `tonal.*`) are
**provisional** until the Essentia-vs-lean-stack spike (vault-tkih) resolves
whether Essentia ships on macOS/ARM or those features come from
librosa/MoSQITo/pyloudnorm. The spike's acceptance includes finalizing these
spellings here. `loudness.*`, `qc.*`, `timbre.*`, `fingerprint.*` do not depend
on Essentia and are stable.
