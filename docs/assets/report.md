# smpl analysis report

**13 frame(s):** 1× audio, 9× feature, 1× image, 1× marker, 1× text

## Features (9 frame(s))

| key | value | unit | role | op |
|---|---|---|---|---|
| `loudness.integrated_lufs` | -20.79 | LUFS | loudness | loudness |
| `loudness.true_peak_dbtp` | -6.71 | dBTP | loudness | loudness |
| `loudness.max_short_term_lufs` | -20.71 | LUFS | loudness | loudness |
| `lowlevel.spectral_flatness_db` | -49.3421 (±22.5402) | dB | spectral | spectral |
| `lowlevel.spectral_crest` | 289.712 (±127.3717) |  | spectral | spectral |
| `lowlevel.spectral_spread` | 3094.8878 (±2873.6923) |  | spectral | spectral |
| `lowlevel.spectral_rolloff` | 5228.3865 (±7381.6941) |  | spectral | spectral |
| `lowlevel.spectral_contrast` | 13.741 (±2.6692) |  | spectral | spectral |
| `lowlevel.spectral_slope` | -0.0001 (±0) |  | spectral | spectral |
| `lowlevel.spectral_skewness` | 16.5459 (±14.9827) |  | spectral | spectral |
| `lowlevel.spectral_kurtosis` | 537.429 (±590.2015) |  | spectral | spectral |
| `qc.clipping.detected` | false |  | qc | qc |
| `qc.dc_offset_dbfs` | -68.9 | dBFS | qc | qc |
| `qc.snr_db` | 13.91 | dB | qc | qc |
| `qc.lossy.spectral_cutoff_hz` | 19993.6 | Hz | qc | qc |
| `qc.lossy.expected_nyquist_hz` | 22050 | Hz | qc | qc |
| `qc.lossy.confidence` | 0.001 | 0–1 | qc | qc |
| `envelope.peak_db_over_floor` | 39.776 | dB | envelope.percussive | envelope |
| `envelope.attack_ms_10_90` | 10.385 | ms | envelope.percussive | envelope |
| `envelope.rise_slope_db_ms` | 1.8262 | dB/ms | envelope.percussive | envelope |
| `envelope.t20_ms` | 66.871 | ms | envelope.percussive | envelope |
| `envelope.early_decay_slope` | -0.2959 | dB/ms | envelope.percussive | envelope |
| `envelope.sustain_ratio_150ms` | 0.0203 | ratio | envelope.percussive | envelope |
| `envelope.peak_db_over_floor` | 10.223 | dB | envelope.sub | envelope |
| `envelope.attack_ms_10_90` | 37.256 | ms | envelope.sub | envelope |
| `envelope.rise_slope_db_ms` | 0.0675 | dB/ms | envelope.sub | envelope |
| `envelope.t20_ms` | — | ms | envelope.sub | envelope |
| `envelope.early_decay_slope` | -0.075 | dB/ms | envelope.sub | envelope |
| `envelope.sustain_ratio_150ms` | 0.5873 | ratio | envelope.sub | envelope |
| `width.sub.correlation` | 1 | −1..1 | width | width |
| `width.sub.side_mid_ratio` | 0 | ratio | width | width |
| `width.bass.correlation` | 1 | −1..1 | width | width |
| `width.bass.side_mid_ratio` | 0 | ratio | width | width |
| `width.lomid.correlation` | 1 | −1..1 | width | width |
| `width.lomid.side_mid_ratio` | 0 | ratio | width | width |
| `width.mid.correlation` | 1 | −1..1 | width | width |
| `width.mid.side_mid_ratio` | 0 | ratio | width | width |
| `width.uppermid.correlation` | 1 | −1..1 | width | width |
| `width.uppermid.side_mid_ratio` | 0 | ratio | width | width |
| `width.air.correlation` | 1 | −1..1 | width | width |
| `width.air.side_mid_ratio` | 0 | ratio | width | width |
| `width.full.correlation` | 1 | −1..1 | width | width |
| `width.full.side_mid_ratio` | 0 | ratio | width | width |
| `movement.sidechain_db` | 10.454 | dB | movement | movement |
| `movement.bass_mod_depth_db` | 26.226 | dB | movement | movement |
| `movement.hf_mod_depth_db` | 42.601 | dB | movement | movement |
| `movement.hf_silence_pct` | 50.43 | % | movement | movement |
| `movement.tail_decay_200ms_db` | 3.473 | dB | movement | movement |
| `clarity.low_mid_masking_db` | 3.381 | dB | clarity | clarity |
| `clarity.mud_presence_ratio` | 0.8913 | ratio | clarity | clarity |
| `clarity.band_contrast_lo_db` | 33.11 | dB | clarity | clarity |
| `clarity.band_contrast_hi_db` | 1.928 | dB | clarity | clarity |
| `clarity.presence_focus_ratio` | 0.004 | ratio | clarity | clarity |
| `clarity.presence_transient_ratio` | 3.773 | crest | clarity | clarity |
| `space.mono_collapse_penalty_db` | 0 | dB | space | space |

## Markers (1 frame(s))

- **defect** — 64 point(s): 0s (click), 0.005s (click), 0.01s (click), 0.0151s (click), 0.0202s (click), … (+59)

## Images (1 frame(s))

| role | media | path |
|---|---|---|
| spectrogram:mel | image/png | `/var/folders/rk/vg49h3t94vv2657_m0j9zdsw0000gn/T/tmp.ItR41c41K5/cas/a6/a62fabc7d611c8e3ce2dd7127cdbc4e4ed2855734b34a86065b49f301081b725.png` |

## Audio (1 frame(s))

- **source** — 4s · 44100 Hz · 1ch · `blake3:dad551e50ae080b8b28bfe45358067f7c8a957abf347dfde5a0a968d79cb9bd9`

## Text / captions (1 frame(s))

- **caption:** 4.00 s · 44100 Hz · 1ch — integrated -20.8 LUFS · true-peak -6.7 dBTP — brightness ~5228 Hz · flatness -49.3 dB — attack 10.4 ms · sustain150 0.02 — QC: low SNR 14 dB

