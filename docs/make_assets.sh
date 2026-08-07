#!/usr/bin/env bash
# Regenerate docs/assets from REAL toolchain runs — the site shows actual smpl
# output, never mockups. Deterministic (seeded) so re-runs diff cleanly.
# Needs: the repo .venv (uv sync), ffmpeg on PATH (for the lossy-origin demo).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY="$REPO/.venv/bin/python"
SMPL="$REPO/.venv/bin/smpl"
SMPL_CAS_DIR="$(mktemp -d)/cas"
export SMPL_CAS_DIR
WORK="$(mktemp -d)"
OUT="$HERE/assets"
mkdir -p "$OUT"

# ---------------------------------------------------------------- sources ----
# A 4-second techno-ish loop (kicks / hats / offbeat bass), bounced QUIET on
# purpose (≈ −21 LUFS) so the normalize demo has something honest to fix; a
# sustained pad for the envelope demo; a bright airy texture for the
# lossy-origin demo (sustained highs make an encoder ceiling visible).
"$PY" - "$WORK/loop.wav" "$WORK/pad.wav" "$WORK/bright.wav" <<'PYEOF'
import sys
import numpy as np
import soundfile as sf

sr = 44100
bpm = 120
beat = 60 / bpm
rng = np.random.default_rng(7)

n = int(sr * beat * 8)
sig = np.zeros(n)
for b in range(8):                                # kicks on quarters
    s = int(b * beat * sr)
    tt = np.arange(min(int(0.5 * sr), n - s)) / sr
    f = 110 * np.exp(-20 * tt) + 48
    sig[s:s + len(tt)] += np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-9 * tt) * 0.9
for h in range(16):                               # hats on eighths
    s = int(h * beat / 2 * sr)
    tt = np.arange(min(int(0.08 * sr), n - s)) / sr
    sig[s:s + len(tt)] += rng.standard_normal(len(tt)) * np.exp(-60 * tt) * 0.25
for b in range(8):                                # offbeat bass line
    s = int((b * beat + beat / 2) * sr)
    tt = np.arange(min(int(0.4 * sr), n - s)) / sr
    f0 = [55, 55, 73.4, 55, 82.4, 55, 73.4, 65.4][b]
    env = np.minimum(tt * 80, 1) * np.exp(-4 * tt)
    sig[s:s + len(tt)] += (np.sin(2 * np.pi * f0 * tt) + 0.4 * np.sin(4 * np.pi * f0 * tt)) * env * 0.5
sig = np.tanh(sig * 0.9) * 0.5
sf.write(sys.argv[1], sig.astype(np.float32), sr, subtype="PCM_16")

n2 = int(sr * 3.0)                                # sustained detuned pad
t = np.arange(n2) / sr
pad = np.zeros(n2)
for f0 in (110.0, 110.7, 164.8, 220.3):
    pad += np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28))
pad *= np.minimum(t * 12, 1) * np.minimum((n2 / sr - t) * 12, 1) * 0.18
sf.write(sys.argv[2], pad.astype(np.float32), sr, subtype="PCM_16")

n3 = int(sr * 4)                                  # airy sustained texture
t3 = np.arange(n3) / sr
rng2 = np.random.default_rng(11)
noise = rng2.standard_normal(n3) * 0.12
chord = sum(np.sin(2 * np.pi * f * t3) * 0.08 for f in (220, 277.2, 329.6, 554.4))
hats2 = np.zeros(n3)
for h in range(16):
    s = int(h * 0.25 * sr)
    tt = np.arange(min(int(0.22 * sr), n3 - s)) / sr
    hats2[s:s + len(tt)] += rng2.standard_normal(len(tt)) * np.exp(-14 * tt) * 0.35
sf.write(sys.argv[3], ((noise + chord + hats2) * 0.5).astype(np.float32), sr, subtype="PCM_16")
print("sources written")
PYEOF

# ------------------------------------------------------------- real pipes ----
loud_json() {  # loudness feature values for a file
  "$SMPL" read "$1" | "$SMPL" loudness 2>/dev/null \
    | "$PY" -c 'import sys,json
for l in sys.stdin:
    f=json.loads(l)
    if f.get("kind")=="feature" and f.get("role")=="loudness":
        print(json.dumps(f["data"]))'
}

echo "-- normalize demo"
"$SMPL" read "$WORK/loop.wav" | "$SMPL" normalize --lufs -14 2>/dev/null \
  | "$SMPL" write "$WORK/leveled.wav" >/dev/null
LOUD_BEFORE="$(loud_json "$WORK/loop.wav")"
LOUD_AFTER="$(loud_json "$WORK/leveled.wav")"

echo "-- high-pass demo"
"$SMPL" read "$WORK/loop.wav" | "$SMPL" filter --hp 200 2>/dev/null \
  | "$SMPL" write "$WORK/thin.wav" >/dev/null

echo "-- envelope demo"
"$SMPL" read "$WORK/pad.wav" | "$SMPL" env --pluck --release 0.6 2>/dev/null \
  | "$SMPL" write "$WORK/pluck.wav" >/dev/null

echo "-- anatomy pipe (frame kinds per stage, for the step-by-step diagram)"
ANATOMY="$( (cd "$WORK" && "$SMPL" read loop.wav | "$SMPL" loudness 2>/dev/null \
  | "$SMPL" qc 2>/dev/null | "$SMPL" view 2>/dev/null) \
  | "$PY" -c 'import sys,json
print(json.dumps([[json.loads(l)["kind"], json.loads(l).get("role","")] for l in sys.stdin]))')"

echo "-- onset markers"
ONSETS="$("$SMPL" read "$WORK/loop.wav" | "$SMPL" slice 2>/dev/null \
  | "$PY" -c 'import sys,json
for l in sys.stdin:
    f=json.loads(l)
    if f.get("kind")=="marker":
        print(json.dumps([round(m["t"],3) for m in f["data"]])); break')"

echo "-- lossy-origin demo (mp3 round trip of the bright texture)"
ffmpeg -loglevel error -y -i "$WORK/bright.wav" -b:a 64k "$WORK/bright.mp3"
ffmpeg -loglevel error -y -i "$WORK/bright.mp3" "$WORK/lossy.wav"
# ONE pipe per file, exactly as shown on the page: qc numbers and the mel
# spectrogram come out of the same run.
cat > "$WORK/qc_spec.py" <<'PYEOF'
import json
import shutil
import subprocess
import sys

out_png, smpl_bin = sys.argv[1], sys.argv[2]
qc = img_hash = None
for line in sys.stdin:
    f = json.loads(line)
    if f.get("kind") == "feature" and f.get("role") == "qc":
        qc = f["data"]
    if f.get("kind") == "image":
        img_hash = f["hash"]
path = subprocess.run([smpl_bin, "resolve", img_hash],
                      capture_output=True, text=True, check=True).stdout.strip()
shutil.copy(path, out_png)
print(json.dumps(qc))
PYEOF
qc_and_spec() {  # in.wav out.png -> prints the qc feature json
  "$SMPL" read "$1" | "$SMPL" qc 2>/dev/null \
    | "$SMPL" spectrogram --kind mel 2>/dev/null \
    | "$PY" "$WORK/qc_spec.py" "$2" "$SMPL"
}
QC_CLEAN="$(qc_and_spec "$WORK/bright.wav" "$OUT/spec-bright.png")"
QC_LOSSY="$(qc_and_spec "$WORK/lossy.wav" "$OUT/spec-lossy.png")"

echo "-- spectrogram of the demo loop (the report figure)"
spec_png() {  # file -> copy the mel PNG out of the CAS
  local h
  h="$("$SMPL" read "$1" | "$SMPL" spectrogram --kind mel 2>/dev/null \
    | "$PY" -c 'import sys,json
for l in sys.stdin:
    f=json.loads(l)
    if f.get("kind")=="image":
        print(f["hash"]); break')"
  cp "$("$SMPL" resolve "$h")" "$2"
}
spec_png "$WORK/loop.wav"   "$OUT/spec-loop.png"

echo "-- memoization timing (same pipe, cold then warm)"
MEMO="$("$PY" - "$SMPL" "$WORK/loop.wav" <<'PYEOF'
import json, subprocess, sys, time
smpl, wav = sys.argv[1], sys.argv[2]
def run():
    t0 = time.time()
    read = subprocess.run([smpl, "read", wav], capture_output=True, check=True)
    subprocess.run([smpl, "spectrogram", "--kind", "mel"], input=read.stdout,
                   capture_output=True, check=True)
    return round(time.time() - t0, 2)
print(json.dumps({"cold_s": run(), "warm_s": run()}))
PYEOF
)"

echo "-- the view report + a real frame"
"$SMPL" read "$WORK/loop.wav" | "$SMPL" describe-all 2>/dev/null \
  | "$SMPL" view 2>/dev/null \
  | "$PY" -c 'import sys,json
for l in sys.stdin:
    f=json.loads(l)
    if f.get("kind")=="text" and f.get("role")=="report":
        print(f["data"]); break' > "$OUT/report.md"
(cd "$WORK" && "$SMPL" read loop.wav) | "$PY" -c 'import sys,json
print(json.dumps(json.loads(sys.stdin.readline()), indent=1))' > "$OUT/frame.json"

# --------------------------------------------------- waveform SVG renderer ----
# The site's own illustration of the audio files real pipes produced (the PNG
# figures above are smpl's artifacts; these SVGs are page graphics of the wavs).
wf() { # in.wav out.svg color [markers_json]
  "$PY" - "$1" "$2" "$3" "${4:-}" <<'PYEOF'
import sys
import json
import numpy as np
import soundfile as sf

wav, out, color = sys.argv[1], sys.argv[2], sys.argv[3]
marks = json.loads(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else []
data, sr = sf.read(wav)
if data.ndim > 1:
    data = data.mean(axis=1)
W, H = 720, 150
seg = max(1, len(data) // W)
mid, amp = H / 2, H / 2 - 6
tops, bots = [], []
for i in range(W):
    c = data[i * seg:(i + 1) * seg]
    if len(c) == 0:
        c = np.zeros(1)
    tops.append(f"{i},{mid - max(0.006, float(c.max())) * amp:.1f}")
    bots.append(f"{i},{mid - min(-0.006, float(c.min())) * amp:.1f}")
path = "M" + " L".join(tops + bots[::-1]) + " Z"
lines = "".join(
    f'<line x1="{t * sr / seg:.1f}" y1="4" x2="{t * sr / seg:.1f}" y2="{H - 4}" '
    f'stroke="#f0b132" stroke-width="1.5" stroke-dasharray="2 4" opacity="0.85"/>'
    for t in marks)
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
       f'preserveAspectRatio="none" role="img">'
       f'<line x1="0" y1="{mid}" x2="{W}" y2="{mid}" stroke="{color}" stroke-width="0.5" opacity="0.35"/>'
       f'<path d="{path}" fill="{color}" fill-opacity="0.55" stroke="{color}" stroke-width="1"/>'
       f'{lines}</svg>')
open(out, "w").write(svg)
print("wrote", out)
PYEOF
}
CHART="#cfe36a"
AMBER="#f0b132"
wf "$WORK/loop.wav"    "$OUT/wf-loop.svg"    "$CHART"
wf "$WORK/leveled.wav" "$OUT/wf-leveled.svg" "$AMBER"
wf "$WORK/thin.wav"    "$OUT/wf-thin.svg"    "$AMBER"
wf "$WORK/pad.wav"     "$OUT/wf-pad.svg"     "$CHART"
wf "$WORK/pluck.wav"   "$OUT/wf-pluck.svg"   "$AMBER"
wf "$WORK/loop.wav"    "$OUT/wf-slice.svg"   "$CHART" "$ONSETS"

# ------------------------------------------------------- audio for players ----
for f in loop leveled thin pad pluck bright lossy; do
  cp "$WORK/$f.wav" "$OUT/$f.wav"
done

# ------------------------------------------------------------ the receipt ----
"$PY" - "$OUT/numbers.json" "$LOUD_BEFORE" "$LOUD_AFTER" "$QC_CLEAN" \
  "$QC_LOSSY" "$ONSETS" "$MEMO" "$ANATOMY" <<'PYEOF'
import json, sys
keys = ["loudness_before", "loudness_after_normalize_lufs_-14",
        "qc_clean", "qc_lossy", "onsets_s", "memo_timing",
        "anatomy_pipe_kinds"]
receipt = {k: json.loads(v) for k, v in zip(keys, sys.argv[2:])}
json.dump(receipt, open(sys.argv[1], "w"), indent=1)
print(json.dumps(receipt, indent=1))
PYEOF

rm -f "$OUT/norm.wav" "$OUT/hp.wav" "$OUT/wf-norm.svg" "$OUT/wf-hp.svg"  # old names

rm -f "$OUT/waveform.png" "$OUT/spectrogram.png"   # the old hand-rolled pair
rm -rf "$WORK"
echo "done → $OUT"
