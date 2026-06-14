export const meta = {
  name: 'smplstream-heavy-fanout',
  description: 'Scaffold the two-tier heavy generators (cloud/transcribe/stems/embed/synth/midi) as separate PATH-discovered uv projects',
  phases: [
    { title: 'Scaffold', detail: '6 heavy tools as isolated uv projects' },
    { title: 'Verify', detail: 'each degrades gracefully without the heavy dep + frame contract' },
  ],
}

const ROOT = '/Users/chronick-mbp/git/smpl'
const PY = `${ROOT}/.venv/bin/python`
const BRIEF = `${ROOT}/.dev/fanout-heavy.md`
const SPEC = '/Users/chronick-mbp/git/vault/music/smplstream/spec.md'
const GENREF = `${ROOT}/tools/smpl-gen` // the canonical two-tier exemplar

const PREAMBLE = `You are scaffolding ONE heavy tool in the smplstream suite as its OWN isolated uv project (two-tier model).
FIRST read ${BRIEF} (the shared brief — two-tier discipline, source-tool contract, HARD RULES), ${SPEC} (the protocol: Source tools, interop seams, kind:midi), and study the EXACT exemplar to mirror: ${GENREF}/pyproject.toml and ${GENREF}/src/smpl_gen/{__init__,cli,backends}.py — copy its structure (pyproject with [project.scripts], a torch/heavy [project.optional-dependencies] extra, [tool.uv.sources] smplstream={path=...,editable=true}, src/<pkg>/{__init__,cli}.py).

HARD RULES: create ONLY new files under your own tools/<dir>/. NEVER edit the core packages, tools/smpl-gen, or another tool. The DEFAULT install must be LIGHT and WORK WITHOUT the heavy dep — on a missing dep/model/binary, emit a clean smplstream error frame (code 'unsupported') to stdout AND a stderr line with the exact install command; NEVER import torch/whisper/demucs/transformers at module top (lazy-import inside run(), guarded with try/except ImportError → the unsupported path). Validate the CLI runs and degrades gracefully using ${PY} (do NOT install the heavy dep — multi-GB). DO NOT run any 'uv' command. Frames: use the smplstream API (read it at ${ROOT}/packages/smplstream/src/smplstream/__init__.py and frames.py). Memoization: op_version must incorporate weights identity; GPU-nondeterministic ops set cacheable:false in a comment/params.

Deliverable: a complete tools/<dir>/ project (pyproject.toml + src/<pkg>/__init__.py + cli.py, and backends.py if it has model management). Report files_created, the verb it surfaces (smpl <verb>), and that --help + the no-heavy-dep degrade path both work.`

const RESULT = {
  type: 'object', additionalProperties: false,
  properties: { tool: { type: 'string' }, verb: { type: 'string' }, files_created: { type: 'array', items: { type: 'string' } },
    degrades_gracefully: { type: 'boolean' }, help_works: { type: 'boolean' }, notes: { type: 'string' } },
  required: ['tool', 'verb', 'files_created', 'degrades_gracefully', 'help_works', 'notes'],
}

const TOOLS = [
  { dir: 'smpl-cloud', spec: `tools/smpl-cloud — 'smpl cloud' source tool (provider APIs: Stable Audio / ElevenLabs / etc). Source-tool contract: accept all three prompt forms (--prompt flag, --prompt - raw stdin, text/role:prompt frame). Key management: env-var-first SMPL_CLOUD_<PROVIDER>_KEY with SMPL_CLOUD_KEY fallback; a 'smpl cloud auth set/list/rm <provider>' writing a 0600 config; env always overrides; NEVER print/log/store keys, redact in params. Without a key OR without the provider SDK installed → emit unsupported error frame + stderr install/auth hint. On success would emit an audio frame (op:cloud). Implement the full CLI + key mgmt + graceful degrade; the actual network call is behind the SDK import (lazy).` },
  { dir: 'smpl-transcribe', spec: `tools/smpl-transcribe — 'smpl transcribe' filter (Whisper speech/lyrics). Lazy-import whisper; without it → unsupported error frame + 'uv tool install smpl-transcribe[whisper]' hint. On success emits text frames (role lyrics) + marker frames (word/segment timestamps with t + sample). Support --format srt|lrc|vtt exporters (these work on already-produced markers without whisper). Passthrough input frames. op_version incorporates the whisper model id.` },
  { dir: 'smpl-stems', spec: `tools/smpl-stems — 'smpl stems' filter (1→many; Demucs / python-audio-separator). Lazy-import the separator; without it → unsupported error frame + install hint. On success resolves the input audio, separates, and emits N audio frames with roles stem:drums/bass/vocals/other/guitar/piano, op 'demucs', op_version incorporating the model+weights id, lineage=[input id], cacheable note re GPU determinism. Degrade path + CLI + --model arg.` },
  { dir: 'smpl-embed', spec: `tools/smpl-embed — 'smpl embed' filter + 'smpl index' (MERT/CLAP vectors + FAISS). Lazy-import transformers/faiss; without them → unsupported error frame + install hint. embed emits vector frames: dim>64 MUST go to CAS as .npy (cas.put_blob(npy_bytes,'application/x-npy')) referenced by hash, NEVER pickle; tag meta.model + meta.dim + meta.dtype. index builds/queries a FAISS index over emitted vectors. op_version incorporates the model id. CLI surfaces both 'embed' and 'index' verbs (argv dispatch like smpl-gen's models).` },
  { dir: 'smpl-synth', spec: `tools/smpl-synth — 'smpl synth' SuperCollider NRT bridge (source + effect). Shells out to sclang/scsynth (a BINARY, not a python dep) in NON-realtime mode: sclang builds an OSC score, scsynth -N renders offline to a soundfile. Without sclang on PATH → unsupported error frame + 'brew install supercollider' hint. As a source: --synthdef/--code + params → audio frame. As an effect: resolve input audio, feed scsynth -N reading it. op 'sc-nrt', op_version = SynthDef source hash + SC version (use a sclang -version style fingerprint). Pure offline op (memoizable). NO realtime/live server (out of scope).` },
  { dir: 'smpl-midi', spec: `tools/smpl-midi — offline MIDI tools surfaced as 'smpl transcribe-midi' (audio→MIDI via basic-pitch) and 'smpl render-midi' (MIDI→audio via fluidsynth -F or SC NRT). Reserve/emit kind:midi frames (a .mid blob in CAS via cas.put_blob(mid_bytes,'audio/midi'), or small event lists inline as data). Lazy-import basic-pitch/fluidsynth; without them → unsupported error frame + install hint. transcribe-midi: audio frame → midi frame (notes). render-midi: midi frame → audio frame. NO realtime MIDI (out of scope). CLI dispatches the two verbs by argv like smpl-gen.` },
]

phase('Scaffold')
const built = await parallel(
  TOOLS.map((t) => () =>
    agent(`${PREAMBLE}\n\n=== YOUR TOOL: ${t.dir} ===\n${t.spec}`, {
      label: `scaffold:${t.dir}`, phase: 'Scaffold', schema: RESULT,
    })
  )
).then((r) => r.filter(Boolean))
log(`Scaffolded ${built.length}/${TOOLS.length}: ${built.map((b) => `${b.tool}(${b.degrades_gracefully ? 'degrades' : 'NO-DEGRADE'})`).join(', ')}`)

phase('Verify')
const VERDICT = {
  type: 'object', additionalProperties: false,
  properties: { tool: { type: 'string' }, two_tier_ok: { type: 'boolean' }, degrades_ok: { type: 'boolean' },
    no_top_level_heavy_import: { type: 'boolean' }, issues: { type: 'array', items: { type: 'string' } }, severity: { enum: ['none', 'low', 'medium', 'high'] } },
  required: ['tool', 'two_tier_ok', 'degrades_ok', 'no_top_level_heavy_import', 'issues', 'severity'],
}
const verdicts = await parallel(
  built.map((b) => () =>
    agent(`Adversarially verify the heavy tool '${b.tool}' at ${ROOT}/tools/${b.tool}.
Read its files and the brief ${BRIEF}. Check, SKEPTICALLY, by actually running ${PY}: (1) NO heavy dep (torch/whisper/demucs/transformers/faiss/basic_pitch) imported at MODULE TOP — only lazily inside run(); grep the source to confirm. (2) Without the heavy dep installed, the CLI runs and emits an 'unsupported' error FRAME (not a traceback, not a hang) + a stderr install hint. (3) --help works. (4) pyproject mirrors the two-tier exemplar (own [project.scripts], heavy dep behind an extra, [tool.uv.sources] smplstream path). (5) for cloud: keys are env-var-first and NEVER printed/logged. Report two_tier_ok/degrades_ok/no_top_level_heavy_import + concrete issues.`,
      { label: `verify:${b.tool}`, phase: 'Verify', schema: VERDICT, agentType: 'code-reviewer' })
  )
).then((r) => r.filter(Boolean))

return {
  built: built.map((b) => ({ tool: b.tool, verb: b.verb, degrades: b.degrades_gracefully })),
  verdicts: verdicts.map((v) => ({ tool: v.tool, two_tier_ok: v.two_tier_ok, degrades_ok: v.degrades_ok, top_level_clean: v.no_top_level_heavy_import, severity: v.severity, issues: v.issues })),
  problems: verdicts.filter((v) => !v.degrades_ok || !v.no_top_level_heavy_import || v.severity === 'high'),
}
