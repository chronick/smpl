#!/usr/bin/env bash
# Generate the README/Pages visual assets from the real toolchain (run AFTER the analysis
# tools land). Produces docs/assets/{waveform.png,spectrogram.png} from a generated sample,
# so the site shows actual smpl output, not mockups.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY="$REPO/.venv/bin/python"
SMPL="$REPO/.venv/bin/smpl"
export SMPL_CAS_DIR="$(mktemp -d)/cas"
WORK="$(mktemp -d)"
mkdir -p "$HERE/assets"

# A short, punchy kick-ish sample: low sine with a fast pitch+amp decay.
"$PY" - "$WORK/kick.wav" <<'PY'
import sys, numpy as np, soundfile as sf
sr = 44100; n = int(sr * 0.6); t = np.arange(n) / sr
f = 120 * np.exp(-18 * t) + 45          # pitch drop
env = np.exp(-7 * t)                    # amplitude decay
body = np.sin(2 * np.pi * np.cumsum(f) / sr) * env
click = (np.random.default_rng(0).standard_normal(n) * np.exp(-400 * t)) * 0.3
sig = np.clip(body + click, -1, 1).astype(np.float32)
sf.write(sys.argv[1], sig, sr, subtype="PCM_16")
PY

# Waveform PNG (matplotlib, Agg) — the "input" visual.
"$PY" - "$WORK/kick.wav" "$HERE/assets/waveform.png" <<'PY'
import sys, numpy as np, soundfile as sf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
data, sr = sf.read(sys.argv[1])
fig, ax = plt.subplots(figsize=(6, 2.4), dpi=140)
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
t = np.arange(len(data)) / sr
ax.plot(t, data, color="#f0b132", lw=0.8)
ax.fill_between(t, data, color="#f0b132", alpha=0.18)
for s in ax.spines.values(): s.set_color("#30363d")
ax.tick_params(colors="#8b949e", labelsize=8); ax.set_xlabel("seconds", color="#8b949e", fontsize=9)
ax.set_yticks([]); ax.set_xlim(0, t[-1])
fig.tight_layout(); fig.savefig(sys.argv[2], facecolor="#0d1117"); print("wrote", sys.argv[2])
PY

# Mel spectrogram PNG via the real `smpl spectrogram` tool — the "output" visual.
if "$SMPL" --help 2>/dev/null | grep -qE "^  spectrogram "; then
  IMG_HASH="$("$SMPL" read "$WORK/kick.wav" | "$SMPL" spectrogram --kind mel 2>/dev/null \
    | "$PY" -c 'import sys,json; [print(json.loads(l)["hash"]) for l in sys.stdin if json.loads(l).get("kind")=="image"]' | head -1)"
  if [ -n "${IMG_HASH:-}" ]; then
    cp "$("$SMPL" resolve "$IMG_HASH")" "$HERE/assets/spectrogram.png"
    echo "wrote $HERE/assets/spectrogram.png (from smpl spectrogram)"
  fi
else
  echo "note: smpl spectrogram not installed yet; spectrogram.png not regenerated"
fi
rm -rf "$WORK"
