#!/usr/bin/env bash
# End-to-end test of the smpl suite "just like a user would":
#   1. Clean install FROM THE GIT REPO into a throwaway venv (uv tool install).
#   2. Two-tier: install the heavy generator into its OWN isolated venv.
#   3. Run each tool in isolation against the sample corpus.
#   4. Run the composable pipes (frames + raw-WAV bridge + jq).
# Forward-compatible: analysis subcommands are exercised only if present, so this passes
# on the foundation and grows coverage as tools land.
#
# Real-library run (e.g. on the mac-mini):  SMPL_E2E_SAMPLES=/path/to/library ./run_e2e.sh
# Fallback: generates a deterministic fixture corpus.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d)"
export SMPL_CAS_DIR="$WORK/cas"
export UV_TOOL_DIR="$WORK/uvtools" UV_TOOL_BIN_DIR="$WORK/uvbin"
PATH="$UV_TOOL_BIN_DIR:$PATH"
PASS=0; FAIL=0; SKIP=0
ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ✗ $1"; FAIL=$((FAIL+1)); }
skip() { echo "  – $1 (skipped)"; SKIP=$((SKIP+1)); }
have() { smpl --help 2>/dev/null | grep -qE "^  $1 "; }

echo "== smpl e2e =="
echo "repo=$REPO work=$WORK"

# --- 1. clean install from the git repo --------------------------------------------
# Install the `smpl` CLI in one shot; its unpublished sibling packages (smplstream,
# smpl-analysis) are supplied as local `--with` sources — the same shape as the README's
# git install, just local. (smplstream has no console scripts, so it is NOT installed
# standalone — `uv tool install` would reject it for having no executables.)
echo "[1] clean install (uv tool install $REPO/packages/smpl + siblings) ..."
if uv tool install "$REPO/packages/smpl" \
     --with "$REPO/packages/smplstream" \
     --with "$REPO/packages/smpl-analysis" >"$WORK/install.log" 2>&1; then
  command -v smpl >/dev/null && ok "smpl installed: $(command -v smpl)" || bad "smpl not on PATH"
else
  bad "uv tool install failed"; tail -5 "$WORK/install.log"; echo "FAIL=$FAIL"; exit 1
fi

# --- 2. two-tier heavy generator into its own venv ---------------------------------
echo "[2] two-tier generator install ..."
if uv tool install "$REPO/tools/smpl-gen" >/dev/null 2>&1; then
  command -v smpl-gen >/dev/null && ok "smpl-gen in isolated venv" || bad "smpl-gen missing"
else
  skip "smpl-gen install"
fi

# --- sample corpus -----------------------------------------------------------------
SAMPLES="${SMPL_E2E_SAMPLES:-}"
if [ -z "$SAMPLES" ]; then
  SAMPLES="$WORK/fixtures"
  "$(command -v python3)" "$REPO/tests/e2e/make_fixtures.py" "$SAMPLES" 2>/dev/null \
    || smpl --help >/dev/null  # fixtures need numpy/soundfile; fall back below if absent
fi
# bash 3.2 (macOS default) has no `mapfile` — read into the array portably.
WAVS=()
while IFS= read -r _f; do WAVS+=("$_f"); done < <(find "$SAMPLES" -maxdepth 2 -iname '*.wav' 2>/dev/null | head -8)
echo "[*] ${#WAVS[@]} sample(s) from $SAMPLES"
[ "${#WAVS[@]}" -gt 0 ] || { bad "no samples to test"; echo "FAIL=$FAIL"; exit 1; }
S0="${WAVS[0]}"

# --- 3. isolation: each tool on one sample -----------------------------------------
echo "[3] tools in isolation ..."
smpl read "$S0" | head -1 | grep -q '"kind":"audio"' && ok "read → audio frame" || bad "read"
smpl read "$S0" | smpl cat | grep -q '"kind":"feature"' && ok "cat → feature frame" || bad "cat"
for t in loudness spectral qc spectrogram convert; do
  if have "$t"; then
    smpl read "$S0" | smpl "$t" >/dev/null 2>&1 && ok "$t" || bad "$t"
  else
    skip "$t"
  fi
done

# --- 4. composable pipes -----------------------------------------------------------
echo "[4] pipes ..."
smpl read "$S0" | smpl as-wav | sox - -t wav - reverb 30 2>/dev/null \
  | smpl from-wav --role wet --derives-from source | grep -q '"op":"from-wav"' \
  && ok "raw-WAV bridge (as-wav | sox | from-wav)" || bad "raw-WAV bridge"
smpl read "$S0" | smpl cat | python3 -c 'import sys,json; assert any(json.loads(l).get("kind")=="text" for l in sys.stdin)' 2>/dev/null \
  && ok "cat caption present" || bad "cat caption"
smpl read "$S0" | smpl resolve --role source | grep -q '\.wav$' && ok "resolve → path" || bad "resolve"
if command -v smpl-gen >/dev/null; then
  smpl gen --prompt "a distorted drum loop" --duration 0.5 | smpl cat | grep -q '"op":"gen"' \
    && ok "gen → cat (source tool + PATH discovery)" || bad "gen pipe"
fi

# --- cleanup -----------------------------------------------------------------------
uv tool uninstall smpl smplstream smpl-gen >/dev/null 2>&1 || true
rm -rf "$WORK"
echo "== done: PASS=$PASS FAIL=$FAIL SKIP=$SKIP =="
[ "$FAIL" -eq 0 ]
